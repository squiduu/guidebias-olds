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
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from utils_v4 import (
    JSDivergence,
    clean_words,
    clear_console,
    filter_wiki,
    get_batch_data,
    get_logger,
    prepare_models_and_tokenizer,
    prepare_neutral_sents,
    prepare_stereo_sents,
)


def finetune(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    """Generate augmented data with stereotype words and save it using the [MASK] token bias.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info(f"Set device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_runner = wandb.init(project=train_args.project, entity="squiduu", name=train_args.run_name)

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare models and tokenizer: {model_args.model_name}")
    guide, trainee, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
    guide.to(device)
    trainee.to(device)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=trainee.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    logger.info("Prepare gender words.")
    with open(file=f"./data/male/male_words_{data_args.num_target_words}.json", mode="r") as male_fp:
        MALE_WORDS = json.load(male_fp)
    MALE_WORDS = MALE_WORDS[: data_args.num_target_words]

    with open(file=f"./data/female/female_words_{data_args.num_target_words}.json", mode="r") as female_fp:
        FEMALE_WORDS = json.load(female_fp)
    FEMALE_WORDS = FEMALE_WORDS[: data_args.num_target_words]

    logger.info("Prepare stereotype words.")
    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREO_WORDS = json.load(ster_fp)
    STEREO_IDS = (
        torch.tensor(tokenizer.convert_tokens_to_ids(clean_words(_words=STEREO_WORDS, tokenizer=tokenizer)))
        .cuda()
        .to(device)
    )

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = filter_wiki(wiki_words=WIKI_WORDS, gender_words=MALE_WORDS + FEMALE_WORDS, stereo_words=STEREO_WORDS)
    WIKI_WORDS = WIKI_WORDS[: data_args.num_wiki_words]

    #
    male_sents = prepare_stereo_sents(gender_words=MALE_WORDS, wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    female_sents = prepare_stereo_sents(gender_words=FEMALE_WORDS, wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    #
    neutral_sents = prepare_neutral_sents(
        gender_words=MALE_WORDS + FEMALE_WORDS, wiki_words=WIKI_WORDS, data_args=data_args
    )
    neutral_sents = neutral_sents[: len(male_sents)]

    #
    dl = DataLoader(
        dataset=[i for i in range(len(male_sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    #
    num_training_steps = int(train_args.num_epochs * len(dl))
    num_warmup_steps = int(num_training_steps * train_args.warmup_proportion)
    logger.info(f"Set lr scheduler with {num_warmup_steps} warm-up steps.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    jsd_runner = JSDivergence(reduction="batchmean")

    #
    for ep in range(1, int(train_args.num_epochs) + 1):
        #
        optimizer.zero_grad()

        #
        dl = tqdm(dl)
        for iter, batch_idx in enumerate(dl):
            #
            male_sents_batch, female_sents_batch, neutral_sents_batch = get_batch_data(
                batch_idx=batch_idx, male_sents=male_sents, female_sents=female_sents, neutral_sents=neutral_sents
            )

            #
            male_inputs = tokenizer(text=male_sents_batch, padding=True, truncation=True, return_tensors="pt")
            female_inputs = tokenizer(text=female_sents_batch, padding=True, truncation=True, return_tensors="pt")
            neutral_inputs = tokenizer(text=neutral_sents_batch, padding=True, truncation=True, return_tensors="pt")

            #
            male_mask_idx = torch.where(male_inputs["input_ids"] == tokenizer.mask_token_id)[1]
            female_mask_idx = torch.where(female_inputs["input_ids"] == tokenizer.mask_token_id)[1]

            #
            for key in male_inputs.keys():
                male_inputs[key] = torch.Tensor.cuda(male_inputs[key], device=device)
                female_inputs[key] = torch.Tensor.cuda(female_inputs[key], device=device)
                neutral_inputs[key] = torch.Tensor.cuda(neutral_inputs[key], device=device)
            male_mask_idx = torch.Tensor.cuda(male_mask_idx, device=device)
            female_mask_idx = torch.Tensor.cuda(female_mask_idx, device=device)

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    guide_neutral_outputs = guide.forward(**neutral_inputs, output_hidden_states=True)
                trainee_male_outputs = trainee.forward(**male_inputs, output_hidden_states=True)
                trainee_female_outputs = trainee.forward(**female_inputs, output_hidden_states=True)
                trainee_neutral_outputs = trainee.forward(**neutral_inputs, output_hidden_states=True)

                #
                freezed_neutral_hidden = guide_neutral_outputs.hidden_states[-1].mean(dim=1)
                male_stereo_hidden = trainee_male_outputs.hidden_states[-1].mean(dim=1)
                female_stereo_hidden = trainee_female_outputs.hidden_states[-1].mean(dim=1)
                tuning_neutral_hidden = trainee_neutral_outputs.hidden_states[-1].mean(dim=1)

                #
                bias_hidden_jsd = jsd_runner.forward(hidden1=male_stereo_hidden, hidden2=female_stereo_hidden)
                #
                bias_hidden_cossim = 1 - F.cosine_similarity(male_stereo_hidden, female_stereo_hidden).mean()

                #
                lm_hidden_kld = F.kl_div(
                    input=F.log_softmax(tuning_neutral_hidden, dim=-1),
                    target=F.softmax(freezed_neutral_hidden, dim=-1),
                    reduction="batchmean",
                )
                lm_hidden_cossim = 1 - F.cosine_similarity(tuning_neutral_hidden, freezed_neutral_hidden).mean()

                #
                male_probs = trainee_male_outputs.logits.flatten(start_dim=0, end_dim=1)[male_mask_idx].softmax(dim=-1)[
                    :, STEREO_IDS
                ]
                female_probs = trainee_female_outputs.logits.flatten(start_dim=0, end_dim=1)[female_mask_idx].softmax(
                    dim=-1
                )[:, STEREO_IDS]

                neutral_probs = (male_probs + female_probs) / 2.0

                bias_logits_jsd = 0.0
                bias_logits_jsd += F.kl_div(
                    input=trainee_male_outputs.logits.flatten(start_dim=0, end_dim=1)[male_mask_idx].log_softmax(
                        dim=-1
                    )[:, STEREO_IDS],
                    target=neutral_probs,
                    reduction="batchmean",
                )
                bias_logits_jsd += F.kl_div(
                    input=trainee_female_outputs.logits.flatten(start_dim=0, end_dim=1)[female_mask_idx].log_softmax(
                        dim=-1
                    )[:, STEREO_IDS],
                    target=neutral_probs,
                    reduction="batchmean",
                )
                bias_logits_jsd = bias_logits_jsd / 2.0

                #
                loss = bias_hidden_jsd + bias_hidden_cossim + bias_logits_jsd + lm_hidden_kld + lm_hidden_cossim

            #
            scaler.scale(loss / train_args.grad_accum_steps).backward()

            #
            scale = scaler.get_scale()
            skip_scheduler = scale != scaler.get_scale()
            if not skip_scheduler:
                scheduler.step()

            #
            if (iter + 1) % train_args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            #
            dl.set_description(
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Loss: {loss:.4f} - Bias Hidden JSD: {bias_hidden_jsd:.4f} - Bias Hidden Cossim: {bias_hidden_cossim:.4f} - Bias Logits JSD: {bias_logits_jsd:.4f} - LM Hidden KLD: {lm_hidden_kld:.4f} - LM Hidden Cossim: {lm_hidden_cossim:.4f}"
            )
            wandb_runner.log(
                {
                    "Total Loss": loss,
                    "Bias Hidden JSD": bias_hidden_jsd,
                    "Bias Hidden Cossim": bias_hidden_cossim,
                    "Bias Logits JSD": bias_logits_jsd,
                    "LM Hidden KLD": lm_hidden_kld,
                    "LM Hidden Cossim": lm_hidden_cossim,
                }
            )

    # after training
    logger.info("Save a fine-tuned model.")
    trainee.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}_seed{train_args.seed}")
    tokenizer.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}_seed{train_args.seed}")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
