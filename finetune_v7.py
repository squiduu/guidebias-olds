import json
from logging import Logger

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from config import DataArguments, ModelArguments, TrainingArguments
from torch.nn.parallel.data_parallel import DataParallel
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_outputs import MaskedLMOutput
from transformers.trainer_utils import set_seed
from utilities_v7 import (
    clear_console,
    get_batch_data,
    get_group_cossim,
    get_hidden,
    get_ids,
    get_inputs,
    get_logger,
    get_logits,
    get_swit_inputs,
    prepare_data,
    prepare_masked_inputs_and_labels,
    prepare_models_and_tokenizer,
    save_checkpoints,
    swit_input_ids,
    to_cuda,
)


def finetune(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    """Generate augmented data with stereotype words and save it using the [MASK] token bias.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info("Set data parallel training.")
    torch.cuda.set_device(train_args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    train_args.world_size = dist.get_world_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_runner = wandb.init(project="swit-debias", entity="squiduu", name=train_args.run_name)

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare top-{data_args.top_k} unmasker: {model_args.model_name}")
    model, tokenizer = prepare_models_and_tokenizer(model_args=model_args)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    model = DataParallel(module=model, output_device=1)
    model.to(device)

    logger.info("Prepare gender words.")
    with open(file=f"./data/male/male_words.json", mode="r") as male_fp:
        M_WORDS = json.load(male_fp)
    M_WORDS = M_WORDS[: data_args.num_target_words]
    with open(file=f"./data/female/female_words.json", mode="r") as female_fp:
        F_WORDS = json.load(female_fp)
    F_WORDS = F_WORDS[: data_args.num_target_words]

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = WIKI_WORDS[: data_args.num_wiki_words]

    #
    m_sents = prepare_data(gender_words=M_WORDS, wiki_words=WIKI_WORDS, tokenizer=tokenizer)
    f_sents = prepare_data(gender_words=F_WORDS, wiki_words=WIKI_WORDS, tokenizer=tokenizer)

    #
    dl = DataLoader(
        dataset=[i for i in range(len(m_sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
    )

    #
    for ep in range(1, int(train_args.num_epochs) + 1):
        #
        optimizer.zero_grad()

        #
        dl = tqdm(dl)
        for iter, batch_idx in enumerate(dl):
            #
            m_sents_batch = get_batch_data(batch_idx=batch_idx, sents=m_sents)
            f_sents_batch = get_batch_data(batch_idx=batch_idx, sents=f_sents)

            #
            m_inputs, m_mask_idx = get_inputs(sents=m_sents_batch, tokenizer=tokenizer)
            f_inputs, f_mask_idx = get_inputs(sents=f_sents_batch, tokenizer=tokenizer)

            #
            m_inputs, m_mask_idx = to_cuda(inputs=m_inputs, mask_idx=m_mask_idx, device=device)
            f_inputs, f_mask_idx = to_cuda(inputs=f_inputs, mask_idx=f_mask_idx, device=device)

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    m_logits = get_logits(model=model, inputs=m_inputs, mask_idx=m_mask_idx)
                    f_logits = get_logits(model=model, inputs=f_inputs, mask_idx=f_mask_idx)

                    #
                    m_ids = get_ids(base_logits=m_logits, rel_logits=f_logits, top_k=data_args.top_k)
                    f_ids = get_ids(base_logits=f_logits, rel_logits=m_logits, top_k=data_args.top_k)

                    #
                    m_input_ids, f_input_ids, m_mask_idx, f_mask_idx = swit_input_ids(
                        m_input_ids=m_inputs["input_ids"],
                        m_mask_idx=m_mask_idx,
                        m_ids=m_ids,
                        f_input_ids=f_inputs["input_ids"],
                        f_mask_idx=f_mask_idx,
                        f_ids=f_ids,
                    )

                    #
                    m_inputs, f_inputs, m_mask_idx, f_mask_idx = get_swit_inputs(
                        m_input_ids=m_input_ids,
                        m_mask_idx=m_mask_idx,
                        f_input_ids=f_input_ids,
                        f_mask_idx=f_mask_idx,
                        tokenizer=tokenizer,
                        device=device,
                    )

                    #
                    m_masked_inputs, m_labels = prepare_masked_inputs_and_labels(
                        inputs=m_inputs, mask_idx=m_mask_idx, tokenizer=tokenizer
                    )
                    f_masked_inputs, f_labels = prepare_masked_inputs_and_labels(
                        inputs=f_inputs, mask_idx=f_mask_idx, tokenizer=tokenizer
                    )

                #
                new_m_mask_idx = torch.where(m_masked_inputs["input_ids"] == tokenizer.mask_token_id)[1].cuda(device)
                new_f_mask_idx = torch.where(f_masked_inputs["input_ids"] == tokenizer.mask_token_id)[1].cuda(device)

                #
                m_outputs: MaskedLMOutput = model.forward(**m_masked_inputs, labels=m_labels, output_hidden_states=True)
                f_outputs: MaskedLMOutput = model.forward(**f_masked_inputs, labels=f_labels, output_hidden_states=True)

                #
                mlm_loss = torch.stack([m_outputs.loss.mean(), f_outputs.loss.mean()]).mean()

                #
                m_out_logits = m_outputs.logits[
                    torch.arange(torch.Tensor.size(m_masked_inputs["input_ids"])[0]), new_m_mask_idx, :
                ].softmax(dim=-1) / torch.norm(
                    m_outputs.logits[
                        torch.arange(torch.Tensor.size(m_masked_inputs["input_ids"])[0]), new_m_mask_idx, :
                    ].softmax(dim=-1)
                )
                f_out_logits = f_outputs.logits[
                    torch.arange(torch.Tensor.size(f_masked_inputs["input_ids"])[0]), new_f_mask_idx, :
                ].softmax(dim=-1) / torch.norm(
                    f_outputs.logits[
                        torch.arange(torch.Tensor.size(f_masked_inputs["input_ids"])[0]), new_f_mask_idx, :
                    ].softmax(dim=-1)
                )

                #
                avg_m_logits = torch.mean(m_out_logits, dim=0)
                avg_f_logits = torch.mean(f_out_logits, dim=0)

                #
                m_logits_cossim = F.cosine_similarity(m_out_logits, avg_m_logits).mean()
                f_logits_cossim = F.cosine_similarity(f_out_logits, avg_f_logits).mean()
                pair_logits_cossim = F.cosine_similarity(m_out_logits, f_out_logits).mean()

                #
                m_hidden = get_hidden(hidden_state=m_outputs.hidden_states)
                f_hidden = get_hidden(hidden_state=f_outputs.hidden_states)

                #
                m_hidden_cossim = get_group_cossim(logits=m_hidden, dim=1)
                f_hidden_cossim = get_group_cossim(logits=f_hidden, dim=1)
                #
                pair_hidden_cossim = F.cosine_similarity(m_hidden, f_hidden).mean()
                #
                cossim_loss = (
                    torch.stack(
                        [
                            m_logits_cossim,
                            f_logits_cossim,
                            pair_logits_cossim,
                            m_hidden_cossim,
                            f_hidden_cossim,
                            pair_hidden_cossim,
                        ]
                    )
                    .mean()
                    .abs()
                )

                #
                m_logits_dist = torch.dist(m_out_logits, avg_m_logits)
                f_logits_dist = torch.dist(f_out_logits, avg_f_logits)
                pair_logits_dist = torch.dist(m_out_logits, f_out_logits)
                #
                avg_m_hidden = torch.mean(m_hidden, dim=0)
                avg_f_hidden = torch.mean(f_hidden, dim=0)
                #
                m_hidden_dist = torch.dist(m_hidden, avg_m_hidden)
                f_hidden_dist = torch.dist(f_hidden, avg_f_hidden)
                pair_hidden_dist = torch.dist(m_hidden, f_hidden)
                #
                dist_loss = torch.stack(
                    [m_logits_dist, f_logits_dist, pair_logits_dist, m_hidden_dist, f_hidden_dist, pair_hidden_dist]
                ).mean()

                #
                batch_loss = mlm_loss + cossim_loss + (10 / dist_loss)

                #
                scaler.scale(batch_loss / train_args.grad_accum_steps).backward()

            #
            if (iter + 1) % train_args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            #
            dl.set_description(
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Batch loss: {batch_loss:.4f} - MLM loss: {mlm_loss:.4f} - Cossim loss: {cossim_loss:.4f} - Dist loss: {dist_loss:.4f}"
            )
            wandb_runner.log(
                {"train_loss": batch_loss, "mlm_loss": mlm_loss, "cossim_loss": cossim_loss, "dist_loss": dist_loss}
            )

        # after training
        logger.info("Save a fine-tuned model.")
        save_checkpoints(model=model, tokenizer=tokenizer, epoch=ep, model_args=model_args, train_args=train_args)


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
