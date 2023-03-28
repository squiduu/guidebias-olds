import os
import logging
import torch
import torch.nn.functional as F
from logging import Logger
from typing import List, Tuple, Union
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaForMaskedLM
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertModel, AlbertForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding
from config import ModelArguments, TrainingArguments


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
    Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
]:
    # get corresponding tokenizer and model class
    if model_args.model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
        guide = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        guide_enc = BertModel.from_pretrained(model_args.model_name_or_path)
        trainee = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        trainee_enc = BertModel.from_pretrained(model_args.model_name_or_path)

        guide.bert = guide_enc
        trainee.bert = trainee_enc

    elif model_args.model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
        guide = RobertaForMaskedLM.from_pretrained(model_args.model_name_or_path)
        guide_enc = RobertaModel.from_pretrained(model_args.model_name_or_path)
        trainee = RobertaForMaskedLM.from_pretrained(model_args.model_name_or_path)
        trainee_enc = RobertaModel.from_pretrained(model_args.model_name_or_path)

        guide.roberta = guide_enc
        trainee.roberta = trainee_enc
    else:
        tokenizer = AlbertTokenizer.from_pretrained(model_args.model_name_or_path)
        guide = AlbertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        guide_enc = AlbertModel.from_pretrained(model_args.model_name_or_path)
        trainee = AlbertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        trainee_enc = AlbertModel.from_pretrained(model_args.model_name_or_path)

        guide.albert = guide_enc
        trainee.albert = trainee_enc

    #
    guide.cuda()
    trainee.cuda()

    return guide, trainee, tokenizer


def filter_wiki(wiki_words: List[str], gender_words: List[str], stereo_words: List[str]):
    filtered = []
    for word in wiki_words:
        if word not in (gender_words + stereo_words):
            filtered.append(word)

    return filtered


def prepare_masked_stereo_sents(
    wiki_words: List[str], stereo_words: List[str], tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]
) -> List[str]:
    sents = []

    for i in range(len(wiki_words)):
        for j in range(len(stereo_words)):
            sents.append(tokenizer.mask_token + " " + wiki_words[i] + " " + stereo_words[j] + " .")

    return sents


def prepare_neutral_sents(
    wiki_words: List[str], tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]
) -> List[str]:
    sents = []

    for i in range(len(wiki_words)):
        for j in range(len(wiki_words)):
            sents.append(tokenizer.mask_token + " " + wiki_words[i] + " " + wiki_words[j] + " .")

    return sents


def get_batch_data(
    batch_idx: torch.LongTensor, stereo_sents: List[str], neutral_sents: List[str]
) -> Tuple[List[str], List[str]]:
    stereo_sents_batch = []
    neutral_sents_batch = []

    for i in batch_idx:
        stereo_sents_batch.append(stereo_sents[torch.Tensor.item(i)])
        neutral_sents_batch.append(neutral_sents[torch.Tensor.item(i)])

    return stereo_sents_batch, neutral_sents_batch


def get_inputs_and_mask_idx(
    stereo_sents: List[str],
    neutral_sents: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
):
    stereo_inputs = tokenizer(text=stereo_sents, padding=True, truncation=True, return_tensors="pt")
    neutral_inputs = tokenizer(text=neutral_sents, padding=True, truncation=True, return_tensors="pt")

    stereo_mask_idx = torch.where(torch.flatten(stereo_inputs["input_ids"]) == tokenizer.mask_token_id)[0]
    neutral_mask_idx = torch.where(torch.flatten(neutral_inputs["input_ids"]) == tokenizer.mask_token_id)[0]

    return stereo_inputs, stereo_mask_idx, neutral_inputs, neutral_mask_idx


def send_to_cuda(
    stereo_inputs: BatchEncoding,
    stereo_mask_idx: torch.LongTensor,
    neutral_inputs: BatchEncoding,
    neutral_mask_idx: torch.LongTensor,
    device: torch.device,
):
    for key in stereo_inputs.keys():
        stereo_inputs[key] = torch.Tensor.cuda(stereo_inputs[key], device=device)
        neutral_inputs[key] = torch.Tensor.cuda(neutral_inputs[key], device=device)

    #
    stereo_mask_idx = torch.Tensor.cuda(stereo_mask_idx, device=device)
    neutral_mask_idx = torch.Tensor.cuda(neutral_mask_idx, device=device)

    return stereo_inputs, stereo_mask_idx, neutral_inputs, neutral_mask_idx


def get_bias_jsd(
    logits: torch.FloatTensor,
    stereo_mask_idx: torch.LongTensor,
    male_ids: List[int],
    female_ids: List[str],
    reduction: str,
):
    male_probs = logits.flatten(start_dim=0, end_dim=1)[stereo_mask_idx].softmax(dim=-1)[:, male_ids]
    female_probs = logits.flatten(start_dim=0, end_dim=1)[stereo_mask_idx].softmax(dim=-1)[:, female_ids]

    neutral_probs = (male_probs + female_probs) / 2.0

    bias_jsd = 0.0
    bias_jsd += F.kl_div(
        input=logits.flatten(start_dim=0, end_dim=1)[stereo_mask_idx].log_softmax(dim=-1)[:, male_ids],
        target=neutral_probs,
        reduction=reduction,
    )
    bias_jsd += F.kl_div(
        input=logits.flatten(start_dim=0, end_dim=1)[stereo_mask_idx].log_softmax(dim=-1)[:, female_ids],
        target=neutral_probs,
        reduction=reduction,
    )

    return bias_jsd


def get_lm_kld(
    guide_logits: torch.LongTensor,
    trainee_logits: torch.LongTensor,
    neutral_mask_idx: torch.LongTensor,
    reduction: str,
):
    guide_neutral_logits = guide_logits.flatten(start_dim=0, end_dim=1)[neutral_mask_idx]
    trainee_neutral_logits = trainee_logits.flatten(start_dim=0, end_dim=1)[neutral_mask_idx]

    lm_kld = F.kl_div(
        input=guide_neutral_logits.log_softmax(dim=-1),
        target=trainee_neutral_logits.softmax(dim=-1),
        reduction=reduction,
    )

    return lm_kld
