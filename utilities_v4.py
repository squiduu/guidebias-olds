import os
import logging
import torch
from logging import Logger
from typing import List, Union, Tuple
from torch.nn.parallel.data_parallel import DataParallel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel, BertForSequenceClassification
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import (
    RobertaForMaskedLM,
    RobertaModel,
    RobertaForSequenceClassification,
)
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertForMaskedLM, AlbertModel, AlbertForSequenceClassification
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
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"e2e_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def prepare_model_and_tokenizer(
    data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments
) -> Tuple[
    Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
]:
    """Download and prepare the pre-trained model and tokenizer.

    Args:
        model_name_or_path (str): A version of pre-trained model.
    """
    if "bert" in model_args.model_name:
        model_class = BertForMaskedLM
        tokenizer_class = BertTokenizer
    elif "roberta" in model_args.model_name:
        model_class = RobertaForMaskedLM
        tokenizer_class = RobertaTokenizer
    else:
        model_class = AlbertForMaskedLM
        tokenizer_class = AlbertTokenizer

    # get tokenizer regardless of model version
    if train_args.use_ckpt:
        tokenizer = tokenizer_class.from_pretrained(data_args.ckpt_dir)
        model = model_class.from_pretrained(data_args.ckpt_dir)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name)
        model = model_class.from_pretrained(model_args.model_name)

    # set DP
    model.cuda()

    return model, tokenizer


def clean_words(words: List[str], tokenizer: BertTokenizer):
    cleaned_words = []
    for word in words:
        if tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id:
            cleaned_words.append(word)

    return cleaned_words


def get_masked_pairs(
    male_words: List[str],
    female_words: List[str],
    wiki_words: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[List[str], List[str]]:
    #
    masked_male_sents = []
    masked_female_sents = []

    # make male and female masked pairs
    for i in range(len(male_words)):
        for j in range(len(wiki_words)):
            masked_male_sents.append(male_words[i] + " " + wiki_words[j] + " " + tokenizer.mask_token + " .")
            masked_female_sents.append(female_words[i] + " " + wiki_words[j] + " " + tokenizer.mask_token + " .")

    return masked_male_sents, masked_female_sents


def get_batch_data(
    batch_idx: torch.tensor,
    male_masked_sents: List[str],
    female_masked_sents: List[str],
):
    #
    male_masked_sents_batch = []
    female_masked_sents_batch = []

    for i in batch_idx:
        male_masked_sents_batch.append(male_masked_sents[torch.Tensor.item(i)])
        female_masked_sents_batch.append(female_masked_sents[torch.Tensor.item(i)])

    return male_masked_sents_batch, female_masked_sents_batch


def get_inputs(
    masked_male_sents: List[str],
    masked_female_sents: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
):
    # tokenize
    male_inputs = tokenizer(text=masked_male_sents, padding=True, truncation=True, return_tensors="pt")
    female_inputs = tokenizer(text=masked_female_sents, padding=True, truncation=True, return_tensors="pt")

    # get [MASK] token indices
    male_mask_idx = torch.where(male_inputs["input_ids"] == tokenizer.mask_token_id)[1]
    female_mask_idx = torch.where(female_inputs["input_ids"] == tokenizer.mask_token_id)[1]

    return male_inputs, female_inputs, male_mask_idx, female_mask_idx


def send_to_cuda(
    male_inputs: BatchEncoding,
    female_inputs: BatchEncoding,
    male_mask_idx: torch.tensor,
    female_mask_idx: torch.tensor,
    device: torch.device,
):
    for key in male_inputs.keys():
        male_inputs[key] = torch.Tensor.cuda(male_inputs[key], device=device)
        female_inputs[key] = torch.Tensor.cuda(female_inputs[key], device=device)

    male_mask_idx = torch.Tensor.cuda(male_mask_idx, device=device)
    female_mask_idx = torch.Tensor.cuda(female_mask_idx, device=device)

    return male_inputs, female_inputs, male_mask_idx, female_mask_idx


def get_stereotype_probs(
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    inputs: BatchEncoding,
    mask_idx: torch.LongTensor,
    stereotype_ids: List[str],
) -> torch.FloatTensor:
    #
    model.eval()
    outputs = model.forward(**inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx, :]
    probs = logits.softmax(dim=-1)
    #
    stereotype_probs = probs[:, stereotype_ids]

    return stereotype_probs


def get_stereotype_ids(
    male_stereotype_probs: torch.FloatTensor,
    female_stereotype_probs: torch.FloatTensor,
    top_k: int,
    stereotype_ids: List[int],
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    #
    male_stereotype_ids = []
    female_stereotype_ids = []

    #  get male stereotype ids
    for batch_idx in range(male_stereotype_probs.size(0)):
        _, male_sub_indices = (male_stereotype_probs[batch_idx] - female_stereotype_probs[batch_idx]).topk(top_k)
        _, male_div_indices2 = (male_stereotype_probs[batch_idx] / female_stereotype_probs[batch_idx]).topk(top_k)

        temp_sub_indices = torch.clone(male_sub_indices).detach()
        temp_div_indices = torch.clone(male_div_indices2).detach()
        for k in range(top_k):
            temp_sub_indices[k] = stereotype_ids[male_sub_indices[k]]
            temp_div_indices[k] = stereotype_ids[male_div_indices2[k]]

        male_stereotype_ids.append(torch.concat([temp_sub_indices, temp_div_indices]))

    # get female stereotype ids
    for batch_idx in range(female_stereotype_probs.size(0)):
        _, female_sub_indices = (female_stereotype_probs[batch_idx] - male_stereotype_probs[batch_idx]).topk(top_k)
        _, female_div_indices = (female_stereotype_probs[batch_idx] / male_stereotype_probs[batch_idx]).topk(top_k)

        temp_sub_indices = torch.clone(female_sub_indices).detach()
        temp_div_indices = torch.clone(female_div_indices).detach()
        for k in range(top_k):
            temp_sub_indices[k] = stereotype_ids[female_sub_indices[k]]
            temp_div_indices[k] = stereotype_ids[female_div_indices[k]]

        female_stereotype_ids.append(torch.concat([temp_sub_indices, temp_div_indices]))

    return torch.stack(male_stereotype_ids), torch.stack(female_stereotype_ids)


def get_loss(
    model: BertForMaskedLM,
    male_inputs: BatchEncoding,
    male_mask_idx: torch.LongTensor,
    male_stereotype_ids: torch.LongTensor,
    female_inputs: BatchEncoding,
    female_mask_idx: torch.LongTensor,
    female_stereotype_ids: torch.LongTensor,
):
    #
    model.train()

    # for male
    outputs = model.forward(**male_inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(male_inputs["input_ids"])[0]), male_mask_idx, :]
    probs = logits.softmax(dim=-1)

    male_delta_probs = []
    for batch_idx in range(male_inputs["input_ids"].size(0)):
        prev_probs = probs[batch_idx, male_stereotype_ids[batch_idx]]
        new_probs = probs[batch_idx, female_stereotype_ids[batch_idx]]
        male_delta_probs.append((prev_probs - new_probs).abs().mean())

    male_delta_probs = torch.stack(male_delta_probs)

    # for female
    outputs = model.forward(**female_inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(female_inputs["input_ids"])[0]), female_mask_idx, :]
    probs = logits.softmax(dim=-1)

    female_delta_probs = []
    for batch_idx in range(female_inputs["input_ids"].size(0)):
        prev_probs = probs[batch_idx, female_stereotype_ids[batch_idx]]
        new_probs = probs[batch_idx, male_stereotype_ids[batch_idx]]
        female_delta_probs.append((prev_probs - new_probs).abs().mean())

    female_delta_probs = torch.stack(female_delta_probs)

    return (male_delta_probs - female_delta_probs).abs().mean()


def overwrite_state_dict(
    model: DataParallel, model_args: ModelArguments
) -> Tuple[
    Union[BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification],
    Union[BertModel, RobertaModel, AlbertModel],
]:
    """Extract and transfer only the trained weights of the layer matching the new model.

    Args:
        trained_model (DebiasRunner): A debiased fine-tuned model.
        model_args (ModelArguments): A parsed model arguments.
    """
    if "bert" in model_args.model_name:
        glue_model_class = BertForSequenceClassification
        seat_model_class = BertModel
    elif "roberta" in model_args.model_name:
        glue_model_class = RobertaForSequenceClassification
        seat_model_class = RobertaModel
    else:
        glue_model_class = AlbertForSequenceClassification
        seat_model_class = AlbertModel

    # get initialized pre-trained model
    glue_model = glue_model_class.from_pretrained(model_args.model_name)
    seat_model = seat_model_class.from_pretrained(model_args.model_name)

    # get only state dict to move to new models
    trained_state_dict = model.module.state_dict()
    glue_state_dict = glue_model.state_dict()
    seat_state_dict = seat_model.state_dict()

    new_glue_state_dict = {k: v for k, v in trained_state_dict.items() if k in glue_state_dict}
    if "bert" in model_args.model_name:
        new_seat_state_dict = {k[5:]: v for k, v in trained_state_dict.items() if k[5:] in seat_state_dict}
    elif "roberta" in model_args.model_name:
        new_seat_state_dict = {k[8:]: v for k, v in trained_state_dict.items() if k[8:] in seat_state_dict}
    else:
        new_seat_state_dict = {k[7:]: v for k, v in trained_state_dict.items() if k[7:] in seat_state_dict}

    # overwrite entries in the existing initialized state dict
    glue_state_dict.update(new_glue_state_dict)
    seat_state_dict.update(new_seat_state_dict)

    # overwrite updated weights
    glue_model.load_state_dict(glue_state_dict)
    seat_model.load_state_dict(seat_state_dict)

    return glue_model, seat_model


def save_checkpoints(
    model: DataParallel,
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    epoch: int,
    model_args: ModelArguments,
    train_args: TrainingArguments,
):
    # get state dict for glue and seat
    glue_model, seat_model = overwrite_state_dict(model=model, model_args=model_args)

    # save for continual training
    PreTrainedModel.save_pretrained(
        self=model.module, save_directory=f"./out/orig_{model_args.model_name}_{train_args.run_name}_ep{epoch}"
    )
    tokenizer.save_pretrained(f"./out/orig_{model_args.model_name}_{train_args.run_name}_ep{epoch}")

    # save for glue
    glue_model.save_pretrained(f"./out/glue_{model_args.model_name}_{train_args.run_name}_ep{epoch}")
    tokenizer.save_pretrained(f"./out/glue_{model_args.model_name}_{train_args.run_name}_ep{epoch}")

    # save for seat
    seat_model.save_pretrained(f"./out/seat_{model_args.model_name}_{train_args.run_name}_ep{epoch}")
    tokenizer.save_pretrained(f"./out/seat_{model_args.model_name}_{train_args.run_name}_ep{epoch}")
