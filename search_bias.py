import json
from logging import Logger

import torch
from config import DataArguments, ModelArguments, TrainingArguments
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import set_seed
from utilities_v7 import (
    clear_console,
    get_batch_data,
    get_inputs,
    get_logger,
    get_logits,
    get_probs,
    prepare_data,
    prepare_models_and_tokenizer,
    to_cuda,
)


def search_bias(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    logger.info(f"Data args: {data_args}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Train args: {train_args}")

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare model and tokenizer
    logger.info(f"Prepare models and tokenizer: {model_args.model_name_or_path}")
    model, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
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

    dl = DataLoader(
        dataset=[i for i in range(len(m_sents))],
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        pin_memory=True,
    )
    dl = tqdm(dl)

    m_bias_sents = {}
    f_bias_sents = {}
    for batch_idx in dl:
        #
        m_sents_batch = get_batch_data(batch_idx=batch_idx, sents=m_sents)
        f_sents_batch = get_batch_data(batch_idx=batch_idx, sents=f_sents)

        #
        m_inputs, m_mask_idx = get_inputs(sents=m_sents_batch, tokenizer=tokenizer)
        f_inputs, f_mask_idx = get_inputs(sents=f_sents_batch, tokenizer=tokenizer)

        #
        m_inputs, m_mask_idx = to_cuda(inputs=m_inputs, mask_idx=m_mask_idx, device=device)
        f_inputs, f_mask_idx = to_cuda(inputs=f_inputs, mask_idx=f_mask_idx, device=device)

        with torch.no_grad():
            #
            m_logits = get_logits(model=model, inputs=m_inputs, mask_idx=m_mask_idx)
            f_logits = get_logits(model=model, inputs=f_inputs, mask_idx=f_mask_idx)

        #
        m_probs = get_probs(base_logits=m_logits, rel_logits=f_logits, top_k=data_args.top_k)
        f_probs = get_probs(base_logits=f_logits, rel_logits=m_logits, top_k=data_args.top_k)

        #
        for i in range(len(m_sents_batch)):
            m_bias_sents[m_sents_batch[i]] = m_probs[i].item()
            f_bias_sents[f_sents_batch[i]] = f_probs[i].item()

    # sort and get stereotype words
    sorted_m_bias_sents = sorted(m_bias_sents.items(), key=lambda item: item[1], reverse=True)
    sorted_f_bias_sents = sorted(f_bias_sents.items(), key=lambda item: item[1], reverse=True)

    # save as .json file
    with open("./out/m_bias_sents.json", mode="w") as fp:
        json.dump(obj=sorted_m_bias_sents, fp=fp)
    with open("./out/f_bias_sents.json", mode="w") as fp:
        json.dump(obj=sorted_f_bias_sents, fp=fp)


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    search_bias(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
