import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import Logger
from typing import List, Tuple, Union
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertModel
from transformers.tokenization_utils_base import BatchEncoding
from config import DataArguments, ModelArguments, TrainingArguments


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


def prepare_models_and_tokenizer(
    model_args: ModelArguments,
) -> Tuple[
    Union[BertModel, RobertaModel, AlbertModel],
    Union[BertModel, RobertaModel, AlbertModel],
    Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
]:
    if "bert" == model_args.model_name:
        tokenizer_class = BertTokenizer
        model_class = BertModel
    elif "roberta" == model_args.model_name:
        tokenizer_class = RobertaTokenizer
        model_class = RobertaModel
    else:
        tokenizer_class = AlbertTokenizer
        model_class = AlbertModel

    # get tokenizer regardless of model version
    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)
    fixed_model = model_class.from_pretrained(model_args.model_name_or_path)
    tuning_model = model_class.from_pretrained(model_args.model_name_or_path)

    #
    fixed_model.cuda()
    tuning_model.cuda()

    return fixed_model, tuning_model, tokenizer


def clean_words(words: List[str], tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]):
    cleaned = []
    for word in words:
        if tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id:
            cleaned.append(word)

    return cleaned


def filter_wiki(wiki_words: List[str], gender_words: List[str], stereo_words: List[str]):
    filtered = []
    for word in wiki_words:
        if word not in (gender_words + stereo_words):
            filtered.append(word)

    return filtered


def prepare_pairs(gender_words: List[str], wiki_words: List[str], stereo_words: List[str]) -> List[str]:
    sents = []
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            for k in range(len(stereo_words)):
                sents.append(gender_words[i] + " " + wiki_words[j] + " " + stereo_words[k] + " .")

    return sents


def prepare_sents(gender_words: List[str], wiki_words: List[str]) -> List[str]:
    sents = []
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            for k in range(len(wiki_words)):
                sents.append(gender_words[i] + " " + wiki_words[j] + " " + wiki_words[k] + " .")

    return sents


def get_batch_data(
    batch_idx: torch.LongTensor, male_sents: List[str], female_sents: List[str], neutral_sents: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    male_batch = []
    female_batch = []
    neutral_batch = []
    for i in batch_idx:
        male_batch.append(male_sents[torch.Tensor.item(i)])
        female_batch.append(female_sents[torch.Tensor.item(i)])
        neutral_batch.append(neutral_sents[torch.Tensor.item(i)])

    return male_batch, female_batch, neutral_batch


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
