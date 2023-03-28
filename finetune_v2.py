import json
from itertools import chain
from logging import Logger

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from config import DataArguments, ModelArguments, TrainingArguments
from torch.nn.parallel.data_parallel import DataParallel
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_outputs import MaskedLMOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from utilities_v2 import (
    clean_words,
    clear_console,
    filter_bias,
    filter_short_sents,
    get_batch_data,
    get_inputs,
    get_logger,
    get_masked_sents,
    get_paired_sents,
    get_prob_ratio,
    get_switched_sents,
    get_unmasked,
    prepare_masked_inputs_and_labels,
    prepare_model_and_tokenizer,
    save_checkpoints,
    send_to_cuda,
    setup_wandb_run,
)
from wandb.sdk.wandb_run import Run


def run_finetune(
    data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger, wandb_run: Run
):
    """Generate augmented data with stereotype words and save it using the [MASK] token bias.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info(f"Data args: {data_args}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Train args: {train_args}")

    logger.info("Set data parallel training.")
    torch.cuda.set_device(train_args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    train_args.world_size = dist.get_world_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare top-{data_args.top_k} unmasker: {model_args.model_name}")
    model, tokenizer = prepare_model_and_tokenizer(data_args=data_args, model_args=model_args, train_args=train_args)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    model = DataParallel(model)
    model.to(device)

    logger.info("Load gender words and sentences.")
    with open(file=f"./data/male/male_words_new.json", mode="r") as fp:
        MALE_WORDS = json.load(fp)
    with open(file=f"./data/male/male_sents.json", mode="r") as fp:
        male_sents = json.load(fp)
    with open(file=f"./data/female/female_words_new.json", mode="r") as fp:
        FEMALE_WORDS = json.load(fp)
    with open(file=f"./data/female/female_sents.json", mode="r") as fp:
        female_sents = json.load(fp)
    logger.info("Load stereotype words.")
    with open(file=f"./data/stereotype/my_stereotype_words.json", mode="r") as fp:
        stereotypes = json.load(fp)
    STEREOTYPES = clean_words(words=stereotypes, tokenizer=tokenizer)

    #
    STEREOTYPE_IDS = tokenizer.convert_tokens_to_ids(STEREOTYPES)

    #
    filtered_male_sents = filter_short_sents(male_sents)
    filtered_female_sents = filter_short_sents(female_sents)

    #
    new_female_sents = get_paired_sents(prev_sents=filtered_male_sents, prev_words=MALE_WORDS, targ_words=FEMALE_WORDS)
    new_male_sents = get_paired_sents(prev_sents=filtered_female_sents, prev_words=FEMALE_WORDS, targ_words=MALE_WORDS)

    #
    MALE_SENTS = filtered_male_sents + new_male_sents
    FEMALE_SENTS = new_female_sents + filtered_female_sents

    #
    male_masked_sents = get_masked_sents(sents=MALE_SENTS, tokenizer=tokenizer)
    female_masked_sents = get_masked_sents(sents=FEMALE_SENTS, tokenizer=tokenizer)

    #
    dataloader = DataLoader(
        [i for i in range(len(male_masked_sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
    )

    #
    num_training_steps = int(train_args.num_epochs * len(dataloader))
    num_warmup_steps = int(num_training_steps * train_args.warmup_proportion)
    logger.info(f"Set learning rate scheduler with {num_warmup_steps} warm-up steps.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    #
    for epoch in range(1, int(train_args.num_epochs) + 1):
        #
        epoch_loss = 0.0

        #
        optimizer.zero_grad()
        #
        dataloader = tqdm(dataloader)
        for iter, batch_i in enumerate(dataloader):
            #
            male_masked_sents_batch, female_masked_sents_batch = get_batch_data(
                batch_i=batch_i, masked_male_sents=male_masked_sents, masked_female_sents=female_masked_sents
            )

            #
            male_inputs_batch, female_inputs_batch, male_mask_idx_batch, female_mask_idx_batch = get_inputs(
                male_masked_sents=male_masked_sents_batch,
                female_masked_sents=female_masked_sents_batch,
                tokenizer=tokenizer,
            )

            #
            male_inputs_batch, female_inputs_batch, male_mask_idx_batch, female_mask_idx_batch = send_to_cuda(
                male_inputs=male_inputs_batch,
                female_inputs=female_inputs_batch,
                male_mask_idx=male_mask_idx_batch,
                female_mask_idx=female_mask_idx_batch,
                device=device,
            )

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    male_unmasked_ids, male_unmasked_probs = get_unmasked(
                        model=model,
                        inputs=male_inputs_batch,
                        mask_idx=male_mask_idx_batch,
                        stereotype_ids=STEREOTYPE_IDS,
                        top_k=data_args.top_k,
                    )
                    female_unmasked_ids, female_unmasked_probs = get_unmasked(
                        model=model,
                        inputs=female_inputs_batch,
                        mask_idx=female_mask_idx_batch,
                        stereotype_ids=STEREOTYPE_IDS,
                        top_k=data_args.top_k,
                    )

                    #
                    male_prob_ratios = get_prob_ratio(
                        model=model,
                        inputs=male_inputs_batch,
                        unmasked_ids1=male_unmasked_ids,
                        unmasked_ids2=female_unmasked_ids,
                        mask_idx1=male_mask_idx_batch,
                        mask_idx2=female_mask_idx_batch,
                        prev_probs=male_unmasked_probs,
                    )
                    female_prob_ratios = get_prob_ratio(
                        model=model,
                        inputs=female_inputs_batch,
                        unmasked_ids1=female_unmasked_ids,
                        unmasked_ids2=male_unmasked_ids,
                        mask_idx1=female_mask_idx_batch,
                        mask_idx2=male_mask_idx_batch,
                        prev_probs=female_unmasked_probs,
                    )

                    #
                    filtered_male_ids, filtered_female_ids = filter_bias(
                        male_prob_ratios=male_prob_ratios, female_prob_ratios=female_prob_ratios
                    )
                    #
                    if len(filtered_male_ids) > 0:
                        switched_male_sents = get_switched_sents(ids=filtered_male_ids, tokenizer=tokenizer)
                    else:
                        switched_male_sents = {}
                    if len(filtered_female_ids) > 0:
                        switched_female_sents = get_switched_sents(ids=filtered_female_ids, tokenizer=tokenizer)
                    else:
                        switched_female_sents = {}
                    #
                    switched_sents = {}
                    for k, v in chain(switched_male_sents.items(), switched_female_sents.items()):
                        switched_sents[k] = v

                    #
                    if len(switched_sents) == 0:
                        continue

                    #
                    masked_inputs, labels = prepare_masked_inputs_and_labels(sents=switched_sents, tokenizer=tokenizer)

                #
                model.train()
                #
                outputs: MaskedLMOutput = model.forward(**masked_inputs, labels=labels)

            #
            scaler.scale(outputs.loss.mean() / train_args.grad_accum_steps).backward()
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
            dataloader.set_description(
                f"Epoch: {epoch}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Train batch loss: {outputs.loss.mean():.4f}"
            )
            wandb_run.log({"train loss": outputs.loss.mean()})

            #
            epoch_loss += outputs.loss.mean()

        # for every epoch
        logger.info(
            f"Epoch: {epoch}/{int(train_args.num_epochs)} - Train epoch loss: {epoch_loss / len(dataloader):.4f}"
        )

    # after training
    logger.info("Save a fine-tuned model.")
    save_checkpoints(model=model, tokenizer=tokenizer, epoch=epoch, model_args=model_args, train_args=train_args)


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    wandb_run = setup_wandb_run(train_args)
    run_finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger, wandb_run=wandb_run)
