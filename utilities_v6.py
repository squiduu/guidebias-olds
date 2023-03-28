import os
import logging
import torch
import random
import copy
from logging import Logger
from typing import List, Tuple
from torch.nn.parallel.data_parallel import DataParallel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel, BertForSequenceClassification
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


def prepare_model_and_tokenizer(model_args: ModelArguments) -> Tuple[BertForMaskedLM, BertTokenizer]:
    # get tokenizer regardless of model version
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)

    # set DP
    model.cuda()

    return model, tokenizer


def get_masked_sents(gender_words: List[str], wiki_words: List[str], tokenizer: BertTokenizer) -> List[str]:
    #
    masked_sents = []

    #
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            masked_sents.append(gender_words[i] + " " + wiki_words[j] + " " + tokenizer.mask_token + " .")

    return masked_sents


def get_batch_data(batch_idx: torch.tensor, sents: List[str]) -> List[str]:
    #
    masked_sents_batch = []

    for i in batch_idx:
        masked_sents_batch.append(sents[torch.Tensor.item(i)])

    return masked_sents_batch


def get_inputs(sents: List[str], tokenizer: BertTokenizer) -> Tuple[BatchEncoding, torch.LongTensor]:
    # tokenize
    inputs = tokenizer(text=sents, padding=True, truncation=True, return_tensors="pt")

    # get [MASK] token indices
    mask_idx = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    return inputs, mask_idx


def send_to_cuda(
    inputs: BatchEncoding, mask_idx: torch.LongTensor, device: torch.device
) -> Tuple[BatchEncoding, torch.LongTensor]:
    for key in inputs.keys():
        inputs[key] = torch.Tensor.cuda(inputs[key], device=device)

    mask_idx = torch.Tensor.cuda(mask_idx, device=device)

    return inputs, mask_idx


def get_probs(model: BertForMaskedLM, inputs: BatchEncoding, mask_idx: torch.LongTensor) -> torch.FloatTensor:
    #
    model.eval()
    outputs = model.forward(**inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx, :]
    probs = logits.softmax(dim=-1)

    return probs


def get_unmasked_ids(probs: torch.FloatTensor, top_k: int) -> torch.LongTensor:
    #
    ids = []

    #  get male stereotype ids
    for batch_idx in range(probs.size(0)):
        _, indices = probs[batch_idx].topk(top_k)

        for k in range(top_k):
            ids.append(indices[k])

    return torch.tensor(list(set(ids)))


def get_switched_input_ids(
    term_input_ids: torch.LongTensor,
    term_mask_ids: torch.LongTensor,
    name_input_ids: torch.LongTensor,
    name_mask_ids: torch.LongTensor,
    term_ids: torch.LongTensor,
    name_ids: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    #
    term_new_input_ids = torch.clone(term_input_ids).detach()
    name_new_input_ids = torch.clone(name_input_ids).detach()

    # for term
    for batch_idx in range(torch.Tensor.size(term_new_input_ids)[0]):
        term_new_input_ids[batch_idx][term_mask_ids[batch_idx]] = random.choice(
            torch.tensor([term_ids[batch_idx], name_ids[batch_idx]])
        )
        name_new_input_ids[batch_idx][name_mask_ids[batch_idx]] = random.choice(
            torch.tensor([term_ids[batch_idx], name_ids[batch_idx]])
        )

    return term_new_input_ids, name_new_input_ids


def get_combined_inputs(
    male_term_input_ids: torch.LongTensor,
    male_name_input_ids: torch.LongTensor,
    male_term_mask_idx: torch.LongTensor,
    male_name_mask_idx: torch.LongTensor,
    female_term_input_ids: torch.LongTensor,
    female_name_input_ids: torch.LongTensor,
    female_term_mask_idx: torch.LongTensor,
    female_name_mask_idx: torch.LongTensor,
    tokenizer: BertTokenizer,
    device: torch.device,
) -> Tuple[BatchEncoding, torch.LongTensor]:
    #
    male_term_sents = tokenizer.batch_decode(male_term_input_ids, skip_special_tokens=True)
    male_name_sents = tokenizer.batch_decode(male_name_input_ids, skip_special_tokens=True)
    female_term_sents = tokenizer.batch_decode(female_term_input_ids, skip_special_tokens=True)
    female_name_sents = tokenizer.batch_decode(female_name_input_ids, skip_special_tokens=True)

    #
    sents = male_term_sents + male_name_sents + female_term_sents + female_name_sents
    inputs = tokenizer(text=sents, padding=True, truncation=True, return_tensors="pt")
    #
    mask_idx = torch.concat([male_term_mask_idx, male_name_mask_idx, female_term_mask_idx, female_name_mask_idx])

    #
    inputs, mask_idx = send_to_cuda(inputs=inputs, mask_idx=mask_idx, device=device)

    return inputs, mask_idx


def prepare_masked_inputs_and_labels(
    inputs: BatchEncoding,
    mask_idx: torch.LongTensor,
    tokenizer: BertTokenizer,
) -> Tuple[BatchEncoding, torch.LongTensor]:
    # copy unmasked inputs as labels
    labels = torch.clone(inputs["input_ids"]).detach()

    # get masked inputs
    masked_inputs = {}
    for key in inputs.keys():
        masked_inputs[key] = torch.clone(inputs[key]).detach()

    for batch_idx in range(torch.Tensor.size(masked_inputs["input_ids"])[0]):
        masked_inputs["input_ids"][batch_idx, mask_idx[batch_idx]] = tokenizer.mask_token_id

    return masked_inputs, labels


def overwrite_state_dict(
    model: DataParallel, model_args: ModelArguments
) -> Tuple[BertForSequenceClassification, BertModel]:
    # get initialized pre-trained model
    glue_model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    seat_model = BertModel.from_pretrained(model_args.model_name_or_path)

    # get only state dict to move to new models
    trained_state_dict = model.module.state_dict()
    glue_state_dict = glue_model.state_dict()
    seat_state_dict = seat_model.state_dict()

    new_glue_state_dict = {k: v for k, v in trained_state_dict.items() if k in glue_state_dict}
    new_seat_state_dict = {k[5:]: v for k, v in trained_state_dict.items() if k[5:] in seat_state_dict}

    # overwrite entries in the existing initialized state dict
    glue_state_dict.update(new_glue_state_dict)
    seat_state_dict.update(new_seat_state_dict)

    # overwrite updated weights
    glue_model.load_state_dict(glue_state_dict)
    seat_model.load_state_dict(seat_state_dict)

    return glue_model, seat_model


def save_checkpoints(
    model: DataParallel, tokenizer: BertTokenizer, epoch: int, model_args: ModelArguments, train_args: TrainingArguments
) -> None:
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
