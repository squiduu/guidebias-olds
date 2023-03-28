import json
from logging import Logger

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import wandb
from config import DataArguments, ModelArguments, TrainingArguments
from torch.nn.parallel.data_parallel import DataParallel
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_outputs import MaskedLMOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from utilities_v6 import (
    clear_console,
    get_batch_data,
    get_combined_inputs,
    get_inputs,
    get_logger,
    get_masked_sents,
    get_probs,
    get_switched_input_ids,
    get_unmasked_ids,
    prepare_masked_inputs_and_labels,
    prepare_model_and_tokenizer,
    save_checkpoints,
    send_to_cuda,
)


def search_bias(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    pass


def run_finetune(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
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

    wandb_runner = wandb.init(project="swit-debias", entity="squiduu")

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare top-{data_args.top_k} unmasker: {model_args.model_name}")
    model, tokenizer = prepare_model_and_tokenizer(model_args)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    model = DataParallel(module=model, output_device=1)
    model.to(device)

    logger.info("Prepare gender terms and names.")
    with open(file=f"./data/male/male_words_{data_args.num_target_words}.json", mode="r") as male_words_fp:
        MALE_TERMS = json.load(male_words_fp)
    with open(file=f"./data/male/male_names_{data_args.num_target_words}.json", mode="r") as male_names_fp:
        MALE_NAMES = json.load(male_names_fp)
    with open(file=f"./data/female/female_words_{data_args.num_target_words}.json", mode="r") as female_words_fp:
        FEMALE_TERMS = json.load(female_words_fp)
    with open(file=f"./data/female/female_names_{data_args.num_target_words}.json", mode="r") as female_names_fp:
        FEMALE_NAMES = json.load(female_names_fp)

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = WIKI_WORDS[: data_args.num_wiki_words]

    #
    male_term_sents = get_masked_sents(gender_words=MALE_TERMS, wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    male_name_sents = get_masked_sents(gender_words=MALE_NAMES, wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    female_term_sents = get_masked_sents(gender_words=FEMALE_TERMS, wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    female_name_sents = get_masked_sents(gender_words=FEMALE_NAMES, wiki_words=WIKI_WORDS, tokenizer=tokenizer)

    #
    dataloader = DataLoader(
        dataset=[i for i in range(len(male_term_sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
    )

    #
    num_warmup_steps = int(train_args.warmup_proportion * len(dataloader))
    num_training_steps = int(train_args.num_epochs * len(dataloader))
    logger.info(f"Set lr scheduler with {num_warmup_steps} warm-up steps.")
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
        for iter, batch_idx in enumerate(dataloader):
            #
            male_term_sents_batch = get_batch_data(batch_idx=batch_idx, sents=male_term_sents)
            male_name_sents_batch = get_batch_data(batch_idx=batch_idx, sents=male_name_sents)
            female_term_sents_batch = get_batch_data(batch_idx=batch_idx, sents=female_term_sents)
            female_name_sents_batch = get_batch_data(batch_idx=batch_idx, sents=female_name_sents)

            #
            male_term_inputs, male_term_mask_idx = get_inputs(sents=male_term_sents_batch, tokenizer=tokenizer)
            male_name_inputs, male_name_mask_idx = get_inputs(sents=male_name_sents_batch, tokenizer=tokenizer)
            female_term_inputs, female_term_mask_idx = get_inputs(sents=female_term_sents_batch, tokenizer=tokenizer)
            female_name_inputs, female_name_mask_idx = get_inputs(sents=female_name_sents_batch, tokenizer=tokenizer)

            #
            male_term_inputs, male_term_mask_idx = send_to_cuda(
                inputs=male_term_inputs, mask_idx=male_term_mask_idx, device=device
            )
            male_name_inputs, male_name_mask_idx = send_to_cuda(
                inputs=male_name_inputs, mask_idx=male_name_mask_idx, device=device
            )
            female_term_inputs, female_term_mask_idx = send_to_cuda(
                inputs=female_term_inputs, mask_idx=female_term_mask_idx, device=device
            )
            female_name_inputs, female_name_mask_idx = send_to_cuda(
                inputs=female_name_inputs, mask_idx=female_name_mask_idx, device=device
            )

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    #
                    male_term_probs = get_probs(model=model, inputs=male_term_inputs, mask_idx=male_term_mask_idx)
                    male_name_probs = get_probs(model=model, inputs=male_name_inputs, mask_idx=male_name_mask_idx)
                    female_term_probs = get_probs(model=model, inputs=female_term_inputs, mask_idx=female_term_mask_idx)
                    female_name_probs = get_probs(model=model, inputs=female_name_inputs, mask_idx=female_name_mask_idx)

                    #
                    male_term_ids = get_unmasked_ids(probs=male_term_probs, top_k=data_args.top_k)
                    male_name_ids = get_unmasked_ids(probs=male_name_probs, top_k=data_args.top_k)
                    female_term_ids = get_unmasked_ids(probs=female_term_probs, top_k=data_args.top_k)
                    female_name_ids = get_unmasked_ids(probs=female_name_probs, top_k=data_args.top_k)

                    #
                    male_term_input_ids, male_name_input_ids = get_switched_input_ids(
                        term_input_ids=male_term_inputs["input_ids"],
                        term_mask_ids=male_term_mask_idx,
                        name_input_ids=male_name_inputs["input_ids"],
                        name_mask_ids=male_name_mask_idx,
                        term_ids=female_term_ids,
                        name_ids=female_name_ids,
                    )
                    female_term_input_ids, female_name_input_ids = get_switched_input_ids(
                        term_input_ids=female_term_inputs["input_ids"],
                        term_mask_ids=female_term_mask_idx,
                        name_input_ids=female_name_inputs["input_ids"],
                        name_mask_ids=female_name_mask_idx,
                        term_ids=male_term_ids,
                        name_ids=male_name_ids,
                    )

                    #
                    inputs, mask_idx = get_combined_inputs(
                        male_term_input_ids=male_term_input_ids,
                        male_name_input_ids=male_name_input_ids,
                        male_term_mask_idx=male_term_mask_idx,
                        male_name_mask_idx=male_name_mask_idx,
                        female_term_input_ids=female_term_input_ids,
                        female_name_input_ids=female_name_input_ids,
                        female_term_mask_idx=female_term_mask_idx,
                        female_name_mask_idx=female_name_mask_idx,
                        tokenizer=tokenizer,
                        device=device,
                    )

                    #
                    masked_inputs, labels = prepare_masked_inputs_and_labels(
                        inputs=inputs, mask_idx=mask_idx, tokenizer=tokenizer
                    )

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
                f"Epoch: {epoch}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Train batch loss: {outputs.loss.mean():.8f}"
            )
            wandb_runner.log({"train_loss": outputs.loss.mean()})

            #
            epoch_loss += outputs.loss.mean()

        # for every epoch
        logger.info(
            f"Epoch: {epoch}/{int(train_args.num_epochs)} - Train epoch loss: {epoch_loss / len(dataloader):.8f}"
        )

        # after training
        logger.info("Save a fine-tuned model.")
        save_checkpoints(model=model, tokenizer=tokenizer, epoch=epoch, model_args=model_args, train_args=train_args)


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    search_bias(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
    run_finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
