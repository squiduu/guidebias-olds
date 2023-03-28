import json
from logging import Logger

import torch
import torch.cuda.amp as amp
import wandb
from config import DataArguments, ModelArguments, TrainingArguments
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from utils_v2 import (
    JSDivergence,
    clear_console,
    filter_wiki,
    get_batch_data,
    get_bias_losses,
    get_inputs_and_mask_idx,
    get_lm_losses,
    get_logger,
    get_mlm_loss,
    prepare_masked_stereo_pairs,
    prepare_models_and_tokenizer,
    prepare_neutral_pairs,
    send_to_cuda,
)


def run_guidebias(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
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
    freezed_model, tuning_model, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
    freezed_model.to(device)
    tuning_model.to(device)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=tuning_model.parameters(), lr=train_args.lr)
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

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = filter_wiki(wiki_words=WIKI_WORDS, gender_words=MALE_WORDS + FEMALE_WORDS, stereo_words=STEREO_WORDS)
    WIKI_WORDS = WIKI_WORDS[: data_args.num_wiki_words]

    #
    male_sents, female_sents = prepare_masked_stereo_pairs(
        male_words=MALE_WORDS,
        female_words=FEMALE_WORDS,
        wiki_words=WIKI_WORDS[:100],
        stereo_words=STEREO_WORDS,
        tokenizer=tokenizer,
    )
    #
    neutral_sents = prepare_neutral_pairs(male_words=MALE_WORDS, female_words=FEMALE_WORDS, wiki_words=WIKI_WORDS)
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
            male_inputs, male_mask_idx, female_inputs, female_mask_idx, neutral_inputs = get_inputs_and_mask_idx(
                male_sents=male_sents_batch,
                female_sents=female_sents_batch,
                neutral_sents=neutral_sents_batch,
                tokenizer=tokenizer,
            )

            #
            male_inputs, male_mask_idx, female_inputs, female_mask_idx, neutral_inputs = send_to_cuda(
                male_inputs=male_inputs,
                male_mask_idx=male_mask_idx,
                female_inputs=female_inputs,
                female_mask_idx=female_mask_idx,
                neutral_inputs=neutral_inputs,
                device=device,
            )

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    freezed_male_outputs = freezed_model.forward(**male_inputs)
                    freezed_female_outputs = freezed_model.forward(**female_inputs)
                    freezed_neutral_outputs = freezed_model.forward(**neutral_inputs, output_hidden_states=True)
                tuning_male_outputs = tuning_model.forward(**male_inputs, output_hidden_states=True)
                tuning_female_outputs = tuning_model.forward(**female_inputs, output_hidden_states=True)
                tuning_neutral_outputs = tuning_model.forward(**neutral_inputs, output_hidden_states=True)

                #
                mlm_loss = get_mlm_loss(
                    freezed_male_outputs=freezed_male_outputs,
                    tuning_male_outputs=tuning_male_outputs,
                    male_mask_idx=male_mask_idx,
                    freezed_female_outputs=freezed_female_outputs,
                    tuning_female_outputs=tuning_female_outputs,
                    female_mask_idx=female_mask_idx,
                )

                #
                bias_jsd, bias_cossim = get_bias_losses(
                    tuning_male_outputs=tuning_male_outputs,
                    tuning_female_outputs=tuning_female_outputs,
                    jsd_runner=jsd_runner,
                )

                #
                lm_kld, lm_cossim = get_lm_losses(
                    tuning_neutral_outputs=tuning_neutral_outputs, freezed_neutral_outputs=freezed_neutral_outputs
                )

                #
                loss = mlm_loss + bias_jsd + bias_cossim + lm_kld + lm_cossim

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
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Loss: {loss:.4f} - MLM Loss: {mlm_loss:.4f} - Bias JSD: {bias_jsd:.4f} - Bias Cossim: {bias_cossim:.4f} - LM KLD: {lm_kld:.4f} - LM Cossim: {lm_cossim:.4f}"
            )
            wandb_runner.log(
                {
                    "Total Loss": loss,
                    "MLM Loss": mlm_loss,
                    "Bias JSD": bias_jsd,
                    "Bias Cossim": bias_cossim,
                    "LM KLD": lm_kld,
                    "LM Cossim": lm_cossim,
                }
            )

        # after training
        logger.info("Save a fine-tuned model.")
        tuning_model.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")
        tokenizer.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    run_guidebias(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
