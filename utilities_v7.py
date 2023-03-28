import os
import logging
import torch
import torch.nn.functional as F
from logging import Logger
from typing import List, Tuple
from torch.nn.parallel.data_parallel import DataParallel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel, BertForSequenceClassification
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


def clean_words(words: List[str], tokenizer: BertTokenizer):
    cleaned_words = []
    for word in words:
        if tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id:
            cleaned_words.append(word)

    return cleaned_words


def prepare_models_and_tokenizer(model_args: ModelArguments) -> Tuple[BertForMaskedLM, BertTokenizer]:
    # get tokenizer regardless of model version
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)

    #
    model.cuda()

    return model, tokenizer


def prepare_data(
    gender_words: List[str], wiki_words: List[str], tokenizer: BertTokenizer
) -> Tuple[List[str], List[int]]:
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


def to_cuda(
    inputs: BatchEncoding, mask_idx: torch.LongTensor, device: torch.device
) -> Tuple[BatchEncoding, torch.LongTensor]:
    for key in inputs.keys():
        inputs[key] = torch.Tensor.cuda(inputs[key], device=device)

    mask_idx = torch.Tensor.cuda(mask_idx, device=device)

    return inputs, mask_idx


def get_logits(model: BertForMaskedLM, inputs: BatchEncoding, mask_idx: torch.LongTensor) -> torch.FloatTensor:
    outputs = model.forward(**inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx, :].softmax(dim=-1)

    return logits


def get_probs(base_logits: torch.FloatTensor, rel_logits: torch.FloatTensor, top_k: int) -> torch.FloatTensor:
    probs, _ = (base_logits - rel_logits).topk(top_k)

    return probs.view(-1)


def get_ids(base_logits: torch.FloatTensor, rel_logits: torch.FloatTensor, top_k: int) -> torch.LongTensor:
    _, indices = (base_logits - rel_logits).topk(top_k)

    return indices.view(-1)


def swit_input_ids(
    m_input_ids: torch.LongTensor,
    m_mask_idx: torch.LongTensor,
    m_ids: torch.LongTensor,
    f_input_ids: torch.LongTensor,
    f_mask_idx: torch.LongTensor,
    f_ids: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    #
    new_m_input_ids = []
    new_f_input_ids = []
    #
    new_m_mask_idx = []
    new_f_mask_idx = []

    #
    for batch_idx in range(torch.Tensor.size(m_input_ids)[0]):
        if m_ids[batch_idx] != -1:
            m_input_ids[batch_idx, m_mask_idx[batch_idx]] = f_ids[batch_idx]

            new_m_input_ids.append(m_input_ids[batch_idx])
            new_m_mask_idx.append(m_mask_idx[batch_idx])

    for batch_idx in range(torch.Tensor.size(f_input_ids)[0]):
        if f_ids[batch_idx] != -1:
            f_input_ids[batch_idx, f_mask_idx[batch_idx]] = m_ids[batch_idx]

            new_f_input_ids.append(f_input_ids[batch_idx])
            new_f_mask_idx.append(f_mask_idx[batch_idx])

    new_m_input_ids = torch.stack(new_m_input_ids)
    new_f_input_ids = torch.stack(new_f_input_ids)
    new_m_mask_idx = torch.stack(new_m_mask_idx)
    new_f_mask_idx = torch.stack(new_f_mask_idx)

    return new_m_input_ids, new_f_input_ids, new_m_mask_idx, new_f_mask_idx


def get_swit_inputs(
    m_input_ids: torch.LongTensor,
    m_mask_idx: torch.LongTensor,
    f_input_ids: torch.LongTensor,
    f_mask_idx: torch.LongTensor,
    tokenizer: BertTokenizer,
    device: torch.device,
) -> Tuple[BatchEncoding, torch.LongTensor]:
    #
    m_sents = tokenizer.batch_decode(m_input_ids, skip_special_tokens=True)
    f_sents = tokenizer.batch_decode(f_input_ids, skip_special_tokens=True)

    #
    m_inputs = tokenizer(text=m_sents, padding=True, truncation=True, return_tensors="pt")
    f_inputs = tokenizer(text=f_sents, padding=True, truncation=True, return_tensors="pt")
    #
    m_inputs, m_mask_idx = to_cuda(inputs=m_inputs, mask_idx=m_mask_idx, device=device)
    f_inputs, f_mask_idx = to_cuda(inputs=f_inputs, mask_idx=f_mask_idx, device=device)

    return m_inputs, f_inputs, m_mask_idx, f_mask_idx


def prepare_masked_inputs_and_labels(
    inputs: BatchEncoding, mask_idx: torch.LongTensor, tokenizer: BertTokenizer
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


def get_hidden(hidden_state: Tuple[torch.FloatTensor]):
    h = hidden_state[-1] / torch.norm(hidden_state[-1])
    hidden = h[:, 1, :]

    return hidden


def get_group_cossim(logits: torch.FloatTensor, dim: int):
    new_logits = logits.clone().detach()

    new_logits[0] = logits[-1]
    new_logits[1:] = logits[:-1]

    return F.cosine_similarity(logits, new_logits, dim=dim).mean()


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
    model: DataParallel, tokenizer: BertTokenizer, model_args: ModelArguments, train_args: TrainingArguments
) -> None:
    # get state dict for glue and seat
    glue_model, seat_model = overwrite_state_dict(model=model, model_args=model_args)

    # save for continual training
    PreTrainedModel.save_pretrained(
        self=model.module, save_directory=f"./out/orig_{model_args.model_name}_{train_args.run_name}"
    )
    tokenizer.save_pretrained(f"./out/orig_{model_args.model_name}_{train_args.run_name}")

    # save for glue
    glue_model.save_pretrained(f"./out/glue_{model_args.model_name}_{train_args.run_name}")
    tokenizer.save_pretrained(f"./out/glue_{model_args.model_name}_{train_args.run_name}")

    # save for seat
    seat_model.save_pretrained(f"./out/seat_{model_args.model_name}_{train_args.run_name}")
    tokenizer.save_pretrained(f"./out/seat_{model_args.model_name}_{train_args.run_name}")
