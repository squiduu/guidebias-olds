import logging
import os
from logging import Logger
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DataArguments, ModelArguments, TrainingArguments
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_logger(train_args: TrainingArguments) -> Logger:
    """Create and set environments for logging.

    Args:
        args (Namespace): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s", datefmt="%Y-%m-%d %H:%M")
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(train_args.output_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"swit_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def prepare_models_and_tokenizer(model_args: ModelArguments) -> Tuple[BertForMaskedLM, BertForMaskedLM, BertTokenizer]:
    # get tokenizer regardless of model version
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    freezed_model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
    freezed_encoder = BertModel.from_pretrained(model_args.model_name_or_path)
    tuning_model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tuning_encoder = BertModel.from_pretrained(model_args.model_name_or_path)

    #
    freezed_model.bert = freezed_encoder
    tuning_model.bert = tuning_encoder

    #
    freezed_model.cuda()
    tuning_model.cuda()

    return freezed_model, tuning_model, tokenizer


def clean_words(_words: List[str], tokenizer: BertTokenizer):
    words = []
    for _word in _words:
        if tokenizer.convert_tokens_to_ids(_words) != tokenizer.unk_token_id:
            words.append(_word)

    return words


def filter_wiki(wiki_words: List[str], gender_words: List[str], stereo_words: List[str]):
    filtered = []
    for word in wiki_words:
        if word not in (gender_words + stereo_words):
            filtered.append(word)

    return filtered


def prepare_stereo_sents(gender_words: List[str], wiki_words: List[str], tokenizer: BertTokenizer) -> List[str]:
    sents = []
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            sents.append(gender_words[i] + " " + wiki_words[j] + " " + tokenizer.mask_token + " .")

    return sents


def prepare_neutral_sents(gender_words: List[str], wiki_words: List[str], data_args: DataArguments) -> List[str]:
    sents = []
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            for k in range(len(wiki_words)):
                sents.append(gender_words[i] + " " + wiki_words[j] + " " + wiki_words[k] + " .")
                if len(sents) >= data_args.num_target_words * data_args.num_wiki_words:
                    break

    return sents


def get_batch_data(
    batch_idx: torch.LongTensor, male_sents: List[str], female_sents: List[str], neutral_sents: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    male_sents_batch = []
    female_sents_batch = []
    neutral_sents_batch = []

    for i in batch_idx:
        male_sents_batch.append(male_sents[torch.Tensor.item(i)])
        female_sents_batch.append(female_sents[torch.Tensor.item(i)])
        neutral_sents_batch.append(neutral_sents[torch.Tensor.item(i)])

    return male_sents_batch, female_sents_batch, neutral_sents_batch


class JSDivergence(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            dim (int, optional): A dimension along which softmax will be computed. Defaults to 1.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, hidden1: torch.FloatTensor, hidden2: torch.FloatTensor) -> torch.FloatTensor:
        h1 = F.softmax(hidden1, dim=1)
        h2 = F.softmax(hidden2, dim=1)

        avg_hidden = (h1 + h2) / 2.0

        jsd = 0.0
        jsd += F.kl_div(input=F.log_softmax(hidden1, dim=1), target=avg_hidden, reduction=self.reduction)
        jsd += F.kl_div(input=F.log_softmax(hidden2, dim=1), target=avg_hidden, reduction=self.reduction)

        return jsd / 2.0
