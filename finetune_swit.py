import json
from logging import Logger

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import wandb
from config import DataArguments, ModelArguments, TrainingArguments
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_outputs import MaskedLMOutput
from transformers.trainer_utils import set_seed
from utilities_swit import (
    clean_words,
    clear_console,
    filter_wiki,
    get_batch_data,
    get_logger,
    prepare_masked_sents,
    prepare_models_and_tokenizer,
)


def finetune(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    """Generate augmented data with stereotype words and save it using the [MASK] token bias.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info(f"Set device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_runner = wandb.init(project=train_args.project, entity="squiduu", name=train_args.run_name)

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare models and tokenizer: {model_args.model_name}")
    model, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
    model.to(device)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    logger.info("Prepare gender words.")
    with open(file=f"./data/male/male_words.json", mode="r") as male_fp:
        M_WORDS = json.load(male_fp)
    M_WORDS = clean_words(words=M_WORDS, tokenizer=tokenizer)
    with open(file=f"./data/female/female_words.json", mode="r") as female_fp:
        F_WORDS = json.load(female_fp)
    F_WORDS = clean_words(words=F_WORDS, tokenizer=tokenizer)

    logger.info("Prepare stereotype words.")
    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREO_WORDS = json.load(ster_fp)

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI = json.load(wiki_fp)
    WIKI = filter_wiki(wiki_words=WIKI, gender_words=M_WORDS + F_WORDS, stereo_words=STEREO_WORDS)
    WIKI = WIKI[: data_args.num_wiki_words]

    NON_GENDER_IDS = [
        i for i in tokenizer.convert_tokens_to_ids(WIKI) if i not in tokenizer.convert_tokens_to_ids(M_WORDS + F_WORDS)
    ]

    #
    male_masked_sents = prepare_masked_sents(gender_words=M_WORDS, wiki_words=WIKI, tokenizer=tokenizer)
    female_masked_sents = prepare_masked_sents(gender_words=F_WORDS, wiki_words=WIKI, tokenizer=tokenizer)

    #
    dl = DataLoader(
        dataset=[i for i in range(len(male_masked_sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    #
    # num_training_steps = int(train_args.num_epochs * len(dl))
    # num_warmup_steps = int(num_training_steps * train_args.warmup_proportion)
    # logger.info(f"Set lr scheduler with {num_warmup_steps} warm-up steps.")
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    # )

    #
    for ep in range(1, int(train_args.num_epochs) + 1):
        #
        optimizer.zero_grad()

        #
        dl = tqdm(dl)
        for iter, batch_idx in enumerate(dl):
            #
            male_masked_sents_batch, female_masked_sents_batch = get_batch_data(
                batch_idx=batch_idx, male_sents=male_masked_sents, female_sents=female_masked_sents
            )

            #
            male_masked_inputs = tokenizer(
                text=male_masked_sents_batch, padding=True, truncation=True, return_tensors="pt"
            )
            female_masked_inputs = tokenizer(
                text=female_masked_sents_batch, padding=True, truncation=True, return_tensors="pt"
            )

            #
            for key in male_masked_inputs.keys():
                male_masked_inputs[key] = torch.Tensor.cuda(male_masked_inputs[key], device=device)
                female_masked_inputs[key] = torch.Tensor.cuda(female_masked_inputs[key], device=device)

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    male_masked_outputs: MaskedLMOutput = model.forward(**male_masked_inputs)
                    female_masked_outputs: MaskedLMOutput = model.forward(**female_masked_inputs)

                #
                male_probs = male_masked_outputs.logits[:, 3, NON_GENDER_IDS].softmax(dim=-1)
                female_probs = female_masked_outputs.logits[:, 3, NON_GENDER_IDS].softmax(dim=-1)

                #
                _, male_idx = (male_probs - female_probs).topk(1)
                _, female_idx = (female_probs - male_probs).topk(1)
                #
                male_tokens = []
                female_tokens = []
                for i in range(male_idx.size(0)):
                    male_tokens.append(tokenizer.convert_ids_to_tokens(NON_GENDER_IDS[male_idx[i]]))
                    female_tokens.append(tokenizer.convert_ids_to_tokens(NON_GENDER_IDS[female_idx[i]]))

                #
                male_unmasked = []
                female_unmasked = []
                for i in range(male_idx.size(0)):
                    male_unmasked.append(str.replace(male_masked_sents_batch[i], tokenizer.mask_token, male_tokens[i]))
                    male_unmasked.append(
                        str.replace(male_masked_sents_batch[i], tokenizer.mask_token, female_tokens[i])
                    )

                    female_unmasked.append(
                        str.replace(female_masked_sents_batch[i], tokenizer.mask_token, male_tokens[i])
                    )
                    female_unmasked.append(
                        str.replace(female_masked_sents_batch[i], tokenizer.mask_token, female_tokens[i])
                    )

                #
                male_inputs = tokenizer(male_unmasked, padding=True, truncation=True, return_tensors="pt")
                female_inputs = tokenizer(female_unmasked, padding=True, truncation=True, return_tensors="pt")

                #
                for key in male_inputs.keys():
                    male_inputs[key] = torch.Tensor.cuda(male_inputs[key], device=device)
                    female_inputs[key] = torch.Tensor.cuda(female_inputs[key], device=device)

                #
                male_outputs = model.forward(**male_inputs, output_hidden_states=True)
                female_outputs = model.forward(**female_inputs, output_hidden_states=True)

                #
                male_cossim = F.cosine_similarity(
                    male_outputs.hidden_states[-1][:, 1, :], male_outputs.hidden_states[-1][:, 3, :]
                )
                female_cossim = F.cosine_similarity(
                    female_outputs.hidden_states[-1][:, 1, :], female_outputs.hidden_states[-1][:, 3, :]
                )
                loss = (male_cossim - female_cossim).abs().mean()

            #
            scaler.scale(loss / train_args.grad_accum_steps).backward()

            #
            # scale = scaler.get_scale()
            # skip_scheduler = scale != scaler.get_scale()
            # if not skip_scheduler:
            #     scheduler.step()

            #
            if (iter + 1) % train_args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            #
            dl.set_description(
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Loss: {loss:.4f}"
            )
            wandb_runner.log({"Loss": loss})

        # for every epoch
        logger.info("Save a fine-tuned model.")
        model.save_pretrained(save_directory=f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")
        tokenizer.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
