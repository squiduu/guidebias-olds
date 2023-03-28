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
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from utilities_v9 import (
    JSDivergence,
    clear_console,
    filter_wiki,
    get_batch_data,
    get_logger,
    prepare_models_and_tokenizer,
    prepare_pairs,
    prepare_sents,
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
    fixed_model, tuning_model, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
    fixed_model.to(device)
    tuning_model.to(device)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=tuning_model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    logger.info("Prepare gender words.")
    with open(file=f"./data/male/male_words_{data_args.num_target_words}.json", mode="r") as male_fp:
        M_WORDS = json.load(male_fp)
    M_WORDS = M_WORDS[: data_args.num_target_words]
    with open(file=f"./data/female/female_words_{data_args.num_target_words}.json", mode="r") as female_fp:
        F_WORDS = json.load(female_fp)
    F_WORDS = F_WORDS[: data_args.num_target_words]

    logger.info("Prepare stereotype words.")
    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREO_WORDS = json.load(ster_fp)

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI = json.load(wiki_fp)
    WIKI = filter_wiki(wiki_words=WIKI, gender_words=M_WORDS + F_WORDS, stereo_words=STEREO_WORDS)
    WIKI = WIKI[: data_args.num_wiki_words]

    #
    male_sents = prepare_pairs(gender_words=M_WORDS, wiki_words=WIKI, stereo_words=STEREO_WORDS)
    female_sents = prepare_pairs(gender_words=F_WORDS, wiki_words=WIKI, stereo_words=STEREO_WORDS)
    #
    neutral_sents = prepare_sents(gender_words=M_WORDS + F_WORDS, wiki_words=WIKI)
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

    #
    for ep in range(1, int(train_args.num_epochs) + 1):
        #
        optimizer.zero_grad()

        #
        dl = tqdm(dl)
        for iter, batch_idx in enumerate(dl):
            #
            male_batch, female_batch, neutral_batch = get_batch_data(
                batch_idx=batch_idx, male_sents=male_sents, female_sents=female_sents, neutral_sents=neutral_sents
            )

            #
            male_inputs = tokenizer(text=male_batch, padding=True, truncation=True, return_tensors="pt")
            female_inputs = tokenizer(text=female_batch, padding=True, truncation=True, return_tensors="pt")
            neutral_inputs = tokenizer(text=neutral_batch, padding=True, truncation=True, return_tensors="pt")

            #
            for key in male_inputs.keys():
                male_inputs[key] = torch.Tensor.cuda(male_inputs[key], device=device)
                female_inputs[key] = torch.Tensor.cuda(female_inputs[key], device=device)
                neutral_inputs[key] = torch.Tensor.cuda(neutral_inputs[key], device=device)

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    fixed_outputs: BaseModelOutputWithPoolingAndCrossAttentions = fixed_model.forward(**neutral_inputs)
                male_outputs: BaseModelOutputWithPoolingAndCrossAttentions = tuning_model.forward(**male_inputs)
                female_outputs: BaseModelOutputWithPoolingAndCrossAttentions = tuning_model.forward(**female_inputs)
                tuning_outputs: BaseModelOutputWithPoolingAndCrossAttentions = tuning_model.forward(**neutral_inputs)

                #
                male_cossim = F.cosine_similarity(
                    male_outputs.last_hidden_state[:, 1, :],
                    male_outputs.last_hidden_state[:, 2:, :].mean(dim=1),
                )
                female_cossim = F.cosine_similarity(
                    female_outputs.last_hidden_state[:, 1, :],
                    female_outputs.last_hidden_state[:, 2:, :].mean(dim=1),
                )
                bias_cossim = (male_cossim - female_cossim).abs().mean()
                #
                lm_kld = F.kl_div(
                    input=F.log_softmax(tuning_outputs.last_hidden_state.mean(dim=1), dim=-1),
                    target=F.softmax(fixed_outputs.last_hidden_state.mean(dim=1), dim=-1),
                    reduction="batchmean",
                )
                lm_cossim = (
                    1
                    - F.cosine_similarity(
                        tuning_outputs.last_hidden_state.mean(dim=1), fixed_outputs.last_hidden_state.mean(dim=1)
                    ).mean()
                )
                #
                loss = bias_cossim + lm_kld + lm_cossim

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
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Loss: {loss:.4f} - Bias Cossim: {bias_cossim:.4f} - LM KLD: {lm_kld:.4f} - LM Cossim: {lm_cossim:.4f}"
            )
            wandb_runner.log(
                {
                    "Total Loss": loss,
                    "Bias Cossim": bias_cossim,
                    "LM KLD": lm_kld,
                    "LM Cossim": lm_cossim,
                }
            )

        # after training
        logger.info("Save a fine-tuned model.")
        tuning_model.save_pretrained(save_directory=f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")
        tokenizer.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
