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
from utils_v3 import (
    clear_console,
    filter_wiki,
    get_batch_data,
    get_bias_jsd,
    get_inputs_and_mask_idx,
    get_lm_kld,
    get_logger,
    prepare_masked_stereo_sents,
    prepare_models_and_tokenizer,
    prepare_neutral_sents,
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
    MALE_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(MALE_WORDS))

    with open(file=f"./data/female/female_words_{data_args.num_target_words}.json", mode="r") as female_fp:
        FEMALE_WORDS = json.load(female_fp)
    FEMALE_WORDS = FEMALE_WORDS[: data_args.num_target_words]
    FEMALE_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(FEMALE_WORDS))

    logger.info("Prepare stereotype words.")
    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREO_WORDS = json.load(ster_fp)

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = filter_wiki(wiki_words=WIKI_WORDS, gender_words=MALE_WORDS + FEMALE_WORDS, stereo_words=STEREO_WORDS)
    WIKI_WORDS = WIKI_WORDS[: data_args.num_wiki_words]

    #
    stereo_masked_sents = prepare_masked_stereo_sents(
        wiki_words=WIKI_WORDS, stereo_words=STEREO_WORDS, tokenizer=tokenizer
    )
    #
    neutral_sents = prepare_neutral_sents(wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    neutral_sents = neutral_sents[: len(stereo_masked_sents)]

    #
    dl = DataLoader(
        dataset=[i for i in range(len(stereo_masked_sents))],
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
            stereo_sents_batch, neutral_sents_batch = get_batch_data(
                batch_idx=batch_idx, stereo_sents=stereo_masked_sents, neutral_sents=neutral_sents
            )

            #
            stereo_inputs, stereo_mask_idx, neutral_inputs, neutral_mask_idx = get_inputs_and_mask_idx(
                stereo_sents=stereo_sents_batch,
                neutral_sents=neutral_sents_batch,
                tokenizer=tokenizer,
            )

            #
            stereo_inputs, stereo_mask_idx, neutral_inputs, neutral_mask_idx = send_to_cuda(
                stereo_inputs=stereo_inputs,
                stereo_mask_idx=stereo_mask_idx,
                neutral_inputs=neutral_inputs,
                neutral_mask_idx=neutral_mask_idx,
                device=device,
            )

            #
            with amp.autocast_mode.autocast():
                trainee_stereo_outputs = trainee.forward(**stereo_inputs)
                trainee_neutral_outputs = trainee.forward(**neutral_inputs)
                with torch.no_grad():
                    guide_neutral_outputs = guide.forward(**neutral_inputs)

                #
                male_ids = torch.Tensor.cuda(MALE_IDS, device=device)
                female_ids = torch.Tensor.cuda(FEMALE_IDS, device=device)

                #
                bias_jsd = get_bias_jsd(
                    logits=trainee_stereo_outputs.logits,
                    stereo_mask_idx=stereo_mask_idx,
                    male_ids=male_ids,
                    female_ids=female_ids,
                    reduction="batchmean",
                )

                #
                lm_kld = get_lm_kld(
                    guide_logits=guide_neutral_outputs.logits,
                    trainee_logits=trainee_neutral_outputs.logits,
                    neutral_mask_idx=neutral_mask_idx,
                    reduction="batchmean",
                )
                #
                loss = bias_jsd + lm_kld

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
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Loss: {loss:.4f} - Bias JSD: {bias_jsd:.4f} - LM KLD: {lm_kld:.4f}"
            )
            wandb_runner.log({"Total Loss": loss, "Bias JSD": bias_jsd, "LM KLD": lm_kld})

        # after training
        logger.info("Save a fine-tuned model.")
        trainee.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}_seed{train_args.seed}")
        tokenizer.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}_seed{train_args.seed}")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    run_guidebias(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
