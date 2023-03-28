import imp
import os
import logging
import random
import torch
import torch.nn as nn
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
from transformers.modeling_outputs import MaskedLMOutput
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
        freezed_model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        freeze_enc = BertModel.from_pretrained(model_args.model_name_or_path)
        tuning_model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        tuning_enc = BertModel.from_pretrained(model_args.model_name_or_path)

        freezed_model.bert = freeze_enc
        tuning_model.bert = tuning_enc

    elif model_args.model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
        freezed_model = RobertaForMaskedLM.from_pretrained(model_args.model_name_or_path)
        freeze_enc = RobertaModel.from_pretrained(model_args.model_name_or_path)
        tuning_model = RobertaForMaskedLM.from_pretrained(model_args.model_name_or_path)
        tuning_enc = RobertaModel.from_pretrained(model_args.model_name_or_path)

        freezed_model.roberta = freeze_enc
        tuning_model.roberta = tuning_enc
    else:
        tokenizer = AlbertTokenizer.from_pretrained(model_args.model_name_or_path)
        freezed_model = AlbertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        freeze_enc = AlbertModel.from_pretrained(model_args.model_name_or_path)
        tuning_model = AlbertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        tuning_enc = AlbertModel.from_pretrained(model_args.model_name_or_path)

        freezed_model.albert = freeze_enc
        tuning_model.albert = tuning_enc

    #
    freezed_model.cuda().eval()
    tuning_model.cuda()

    return freezed_model, tuning_model, tokenizer


def filter_wiki(wiki_words: List[str], gender_words: List[str], stereo_words: List[str]):
    filtered = []
    for word in wiki_words:
        if word not in (gender_words + stereo_words):
            filtered.append(word)

    return filtered


def prepare_masked_stereo_pairs(
    male_words: List[str],
    female_words: List[str],
    wiki_words: List[str],
    stereo_words: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> List[str]:
    male_sents = []
    female_sents = []

    for i in range(len(male_words)):
        for j in range(len(wiki_words)):
            for k in range(len(stereo_words)):

                rand_no = random.choice([0, 1, 2, 3])

                if rand_no == 0:
                    male_sents.append(
                        tokenizer.mask_token + " " + male_words[i] + " " + wiki_words[j] + " " + stereo_words[k] + " ."
                    )
                    female_sents.append(
                        tokenizer.mask_token
                        + " "
                        + female_words[i]
                        + " "
                        + wiki_words[j]
                        + " "
                        + stereo_words[k]
                        + " ."
                    )

                elif rand_no == 1:
                    male_sents.append(
                        male_words[i] + " " + tokenizer.mask_token + " " + wiki_words[j] + " " + stereo_words[k] + " ."
                    )
                    female_sents.append(
                        female_words[i]
                        + " "
                        + tokenizer.mask_token
                        + " "
                        + wiki_words[j]
                        + " "
                        + stereo_words[k]
                        + " ."
                    )

                elif rand_no == 2:
                    male_sents.append(
                        male_words[i] + " " + wiki_words[j] + " " + tokenizer.mask_token + " " + stereo_words[k] + " ."
                    )
                    female_sents.append(
                        female_words[i]
                        + " "
                        + wiki_words[j]
                        + " "
                        + tokenizer.mask_token
                        + " "
                        + stereo_words[k]
                        + " ."
                    )

                else:
                    male_sents.append(
                        male_words[i] + " " + wiki_words[j] + " " + stereo_words[k] + " " + tokenizer.mask_token + " ."
                    )
                    female_sents.append(
                        female_words[i]
                        + " "
                        + wiki_words[j]
                        + " "
                        + stereo_words[k]
                        + " "
                        + tokenizer.mask_token
                        + " ."
                    )

    return male_sents, female_sents


def prepare_neutral_pairs(male_words: List[str], female_words: List[str], wiki_words: List[str]) -> List[str]:
    pairs = []

    for i in range(len(male_words)):
        for j in range(len(wiki_words)):
            for k in range(len(wiki_words)):
                male_sent = male_words[i] + " " + wiki_words[j] + " " + wiki_words[k] + " ."
                female_sent = female_words[i] + " " + wiki_words[j] + " " + wiki_words[k] + " ."

                pairs.append([male_sent, female_sent])

    return pairs


def get_batch_data(
    batch_idx: torch.LongTensor, male_sents: List[str], female_sents: List[str], neutral_sents: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    male_sents_batch = []
    female_sents_batch = []
    temp_neutral_sents = []
    neutral_sents_batch = []

    for i in batch_idx:
        male_sents_batch.append(male_sents[torch.Tensor.item(i)])
        female_sents_batch.append(female_sents[torch.Tensor.item(i)])
        temp_neutral_sents.append(neutral_sents[torch.Tensor.item(i)])

    for neutral_pair in temp_neutral_sents:
        for neutral_sent in neutral_pair:
            neutral_sents_batch.append(neutral_sent)

    return male_sents_batch, female_sents_batch, neutral_sents_batch


def get_inputs_and_mask_idx(
    male_sents: List[str],
    female_sents: List[str],
    neutral_sents: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
):
    male_inputs = tokenizer(text=male_sents, padding=True, truncation=True, return_tensors="pt")
    female_inputs = tokenizer(text=female_sents, padding=True, truncation=True, return_tensors="pt")
    neutral_inputs = tokenizer(text=neutral_sents, padding=True, truncation=True, return_tensors="pt")

    male_mask_idx = torch.where(torch.flatten(male_inputs["input_ids"]) == tokenizer.mask_token_id)[0]
    female_mask_idx = torch.where(torch.flatten(female_inputs["input_ids"]) == tokenizer.mask_token_id)[0]

    return male_inputs, male_mask_idx, female_inputs, female_mask_idx, neutral_inputs


def send_to_cuda(
    male_inputs: BatchEncoding,
    male_mask_idx: torch.LongTensor,
    female_inputs: BatchEncoding,
    female_mask_idx: torch.LongTensor,
    neutral_inputs: BatchEncoding,
    device: torch.device,
):
    for key in male_inputs.keys():
        male_inputs[key] = torch.Tensor.cuda(male_inputs[key], device=device)
        female_inputs[key] = torch.Tensor.cuda(female_inputs[key], device=device)
        neutral_inputs[key] = torch.Tensor.cuda(neutral_inputs[key], device=device)
    #
    male_mask_idx = torch.Tensor.cuda(male_mask_idx, device=device)
    female_mask_idx = torch.Tensor.cuda(female_mask_idx, device=device)

    return male_inputs, male_mask_idx, female_inputs, female_mask_idx, neutral_inputs


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


def get_mlm_loss(
    freezed_male_outputs: MaskedLMOutput,
    tuning_male_outputs: MaskedLMOutput,
    male_mask_idx: torch.LongTensor,
    freezed_female_outputs: MaskedLMOutput,
    tuning_female_outputs: MaskedLMOutput,
    female_mask_idx: torch.LongTensor,
):
    freezed_male_logits = freezed_male_outputs.logits.flatten(start_dim=0, end_dim=1)[male_mask_idx]
    freezed_female_logits = freezed_female_outputs.logits.flatten(start_dim=0, end_dim=1)[female_mask_idx]
    freezed_stereo_logits = (freezed_male_logits + freezed_female_logits) / 2.0

    tuning_male_logits = tuning_male_outputs.logits.flatten(start_dim=0, end_dim=1)[male_mask_idx]
    tuning_female_logits = tuning_female_outputs.logits.flatten(start_dim=0, end_dim=1)[female_mask_idx]

    male_kld = F.kl_div(
        input=F.log_softmax(tuning_male_logits, dim=-1),
        target=F.softmax(freezed_stereo_logits, dim=-1),
        reduction="batchmean",
    )
    female_kld = F.kl_div(
        input=F.log_softmax(tuning_female_logits, dim=-1),
        target=F.softmax(freezed_stereo_logits, dim=-1),
        reduction="batchmean",
    )

    return (male_kld + female_kld) / 2.0


def get_bias_losses(
    tuning_male_outputs: MaskedLMOutput,
    tuning_female_outputs: MaskedLMOutput,
    jsd_runner: JSDivergence,
):
    tuning_male_hidden = tuning_male_outputs.hidden_states[-1].mean(dim=1)
    tuning_female_hidden = tuning_female_outputs.hidden_states[-1].mean(dim=1)

    bias_jsd = jsd_runner.forward(hidden1=tuning_male_hidden, hidden2=tuning_female_hidden)
    bias_cossim = 1 - F.cosine_similarity(tuning_male_hidden, tuning_female_hidden).mean()

    return bias_jsd, bias_cossim


def get_lm_losses(tuning_neutral_outputs: MaskedLMOutput, freezed_neutral_outputs: MaskedLMOutput):
    lm_kld = F.kl_div(
        input=F.log_softmax(tuning_neutral_outputs.hidden_states[-1].mean(dim=1), dim=-1),
        target=F.softmax(freezed_neutral_outputs.hidden_states[-1].mean(dim=1), dim=-1),
        reduction="batchmean",
    )
    lm_cossim = (
        1
        - F.cosine_similarity(
            tuning_neutral_outputs.hidden_states[-1].mean(dim=1), freezed_neutral_outputs.hidden_states[-1].mean(dim=1)
        ).mean()
    )

    return lm_kld, lm_cossim
