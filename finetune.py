import json
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import wandb
from itertools import chain
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data.dataloader import DataLoader
from logging import Logger
from torch.optim.adamw import AdamW
from transformers.trainer_utils import set_seed
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_outputs import MaskedLMOutput
from config import DataArguments, ModelArguments, TrainingArguments
from utilities import (
    clear_console,
    filter_bias,
    get_logger,
    get_prob_ratio,
    get_rev_masked_sents,
    prepare_model_and_tokenizer,
    get_masked_pairs,
    get_batch_data,
    get_inputs,
    send_to_cuda,
    get_stereotype_probs,
    get_switched_sents,
    prepare_masked_inputs_and_labels,
    save_checkpoints,
)


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
    model, tokenizer = prepare_model_and_tokenizer(data_args=data_args, model_args=model_args, train_args=train_args)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    model = DataParallel(module=model, output_device=1)
    model.to(device)

    logger.info("Prepare gender target words.")
    with open(file=f"./data/male/male_words_new.json", mode="r") as male_fp:
        MALE_WORDS = json.load(male_fp)
    MALE_IDS = tokenizer.convert_tokens_to_ids(MALE_WORDS)
    with open(file=f"./data/female/female_words_new.json", mode="r") as female_fp:
        FEMALE_WORDS = json.load(female_fp)
    FEMALE_IDS = tokenizer.convert_tokens_to_ids(FEMALE_WORDS)
    FEMALE_IDS = tokenizer.convert_tokens_to_ids(FEMALE_WORDS)
    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREOTYPE_WORDS = json.load(ster_fp)
    logger.info("Load wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)

    #
    male_masked_sents, female_masked_sents = get_masked_pairs(
        male_words=MALE_WORDS, female_words=FEMALE_WORDS, wiki_words=WIKI_WORDS, tokenizer=tokenizer
    )

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
            male_masked_sents_batch, female_masked_sents_batch = get_batch_data(
                batch_idx=batch_idx, male_masked_sents=male_masked_sents, female_masked_sents=female_masked_sents
            )

            #
            male_inputs_batch, female_inputs_batch, male_mask_idx_batch, female_mask_idx_batch = get_inputs(
                masked_male_sents=male_masked_sents_batch,
                masked_female_sents=female_masked_sents_batch,
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
                    #
                    male_unmasked_ids, male_prev_probs = get_stereotype_probs(
                        model=model,
                        inputs=male_inputs_batch,
                        mask_idx=male_mask_idx_batch,
                        top_k=data_args.top_k,
                        reversed=False,
                        gender_ids=MALE_IDS,
                    )
                    female_unmasked_ids, female_prev_probs = get_stereotype_probs(
                        model=model,
                        inputs=female_inputs_batch,
                        mask_idx=female_mask_idx_batch,
                        top_k=data_args.top_k,
                        reversed=False,
                        gender_ids=FEMALE_IDS,
                    )

                    #
                    male_rev_masked_sents = get_rev_masked_sents(unmasked_ids=male_unmasked_ids, tokenizer=tokenizer)
                    female_rev_masked_sents = get_rev_masked_sents(unmasked_ids=male_unmasked_ids, tokenizer=tokenizer)

                    #
                    male_rev_inputs, female_rev_inputs, male_rev_mask_idx, female_rev_mask_idx = get_inputs(
                        masked_male_sents=male_rev_masked_sents,
                        masked_female_sents=female_rev_masked_sents,
                        tokenizer=tokenizer,
                    )

                    #
                    male_rev_unmasked_ids, male_rev_prev_probs = get_stereotype_probs(
                        model=model,
                        inputs=male_rev_inputs,
                        mask_idx=male_rev_mask_idx,
                        top_k=data_args.top_k,
                        reversed=True,
                        gender_ids=MALE_IDS,
                    )
                    female_rev_unmasked_ids, female_rev_prev_probs = get_stereotype_probs(
                        model=model,
                        inputs=female_rev_inputs,
                        mask_idx=female_rev_mask_idx,
                        top_k=data_args.top_k,
                        reversed=True,
                        gender_ids=FEMALE_IDS,
                    )

                    #
                    male_prob_ratios = get_prob_ratio(
                        model=model,
                        inputs=male_inputs_batch,
                        unmasked_ids1=male_unmasked_ids,
                        unmasked_ids2=female_unmasked_ids,
                        mask_idx1=male_mask_idx_batch,
                        mask_idx2=female_mask_idx_batch,
                        prev_probs=male_prev_probs,
                    )
                    female_prob_ratios = get_prob_ratio(
                        model=model,
                        inputs=female_inputs_batch,
                        unmasked_ids1=female_unmasked_ids,
                        unmasked_ids2=male_unmasked_ids,
                        mask_idx1=female_mask_idx_batch,
                        mask_idx2=male_mask_idx_batch,
                        prev_probs=female_prev_probs,
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

                    # s
                    if len(switched_sents) == 0:
                        continue

                    #
                    masked_inputs, labels = prepare_masked_inputs_and_labels(
                        switched_sents=switched_sents, tokenizer=tokenizer, model_args=model_args
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
                f"Epoch: {epoch}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Train batch loss: {outputs.loss.mean():.4f}"
            )
            wandb_runner.log({"train_loss": outputs.loss.mean()})

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

    run_finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
