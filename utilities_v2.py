import os
import logging
import torch
import wandb
from logging import Logger
from typing import List, Dict, Union, Tuple
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
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"e2e_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def setup_wandb_run(train_args: TrainingArguments):
    if train_args.local_rank == 0:
        wandb_run = wandb.init(project=train_args.project)
    else:
        wandb_run = None

    return wandb_run


def prepare_model_and_tokenizer(
    data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments
) -> Tuple[BertForMaskedLM, BertTokenizer,]:
    """Download and prepare the pre-trained model and tokenizer.

    Args:
        model_name_or_path (str): A version of pre-trained model.
    """
    # get tokenizer regardless of model version
    if train_args.use_ckpt:
        tokenizer = BertTokenizer.from_pretrained(data_args.ckpt_dir)
        model = BertForMaskedLM.from_pretrained(data_args.ckpt_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name)
        model = BertForMaskedLM.from_pretrained(model_args.model_name)

    # set DP
    model.cuda()

    return model, tokenizer


def clean_words(words: List[str], tokenizer: BertTokenizer):
    cleaned_words = []
    for word in words:
        if tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id:
            cleaned_words.append(word)

    return cleaned_words


def filter_short_sents(sents: List[str]):
    filtered_sents = []
    for sent in sents:
        if len(sent.split()) >= 5:
            filtered_sents.append(sent)

    return filtered_sents


def get_paired_sents(prev_sents: List[str], prev_words: List[str], targ_words: List[str]) -> List[str]:
    new_sents = []
    for sent in prev_sents:

        new_str = ""
        for char in sent.split():
            if char in prev_words:
                i = prev_words.index(char)
                new_str += targ_words[i] + " "

            elif char in targ_words:
                i = targ_words.index(char)
                new_str += prev_words[i] + " "

            else:
                new_str += char + " "

        new_sents.append(new_str.strip())

    return new_sents


def get_masked_sents(sents: List[str], tokenizer: BertTokenizer):
    masked_sents = []
    for sent in sents:
        splits = sent.split()
        splits[2] = tokenizer.mask_token
        masked_sents.append(" ".join(splits))

    return masked_sents


def get_batch_data(
    batch_i: torch.tensor,
    masked_male_sents: List[str],
    masked_female_sents: List[str],
):
    #
    male_masked_sents_batch = []
    female_masked_sents_batch = []

    for i in batch_i:
        male_masked_sents_batch.append(masked_male_sents[torch.Tensor.item(i)])
        female_masked_sents_batch.append(masked_female_sents[torch.Tensor.item(i)])

    return male_masked_sents_batch, female_masked_sents_batch


def get_inputs(male_masked_sents: List[str], female_masked_sents: List[str], tokenizer: BertTokenizer):
    # tokenize
    male_inputs = tokenizer(text=male_masked_sents, padding=True, truncation=True, return_tensors="pt")
    female_inputs = tokenizer(text=female_masked_sents, padding=True, truncation=True, return_tensors="pt")

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


def get_unmasked(
    model: BertForMaskedLM, inputs: BatchEncoding, mask_idx: torch.LongTensor, stereotype_ids: List[int], top_k: int
) -> Union[torch.LongTensor, torch.FloatTensor]:
    #
    model.eval()
    outputs = model.forward(**inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx, :]
    probs = logits.softmax(dim=-1)
    # pred_probs, indices = probs.topk(top_k)
    stereotype_probs = probs[:, stereotype_ids]
    pred_probs, indices = stereotype_probs.topk(top_k)

    #
    unmasked_ids = torch.clone(inputs["input_ids"]).detach()
    unmasked_old_probs = []
    for batch_i in range(indices.size(0)):
        for token_idx in range(1):
            unmasked_ids[batch_i][mask_idx[batch_i]] = stereotype_ids[indices[batch_i][token_idx]]
            # unmasked_ids[batch_i][mask_idx[batch_i]] = indices[batch_i][token_idx]
            unmasked_old_probs.append(pred_probs[batch_i].item())

    return unmasked_ids, torch.tensor(unmasked_old_probs)


def get_prob_ratio(
    model: BertForMaskedLM,
    inputs: BatchEncoding,
    unmasked_ids1: torch.LongTensor,
    unmasked_ids2: torch.LongTensor,
    mask_idx1: torch.LongTensor,
    mask_idx2: torch.LongTensor,
    prev_probs: torch.FloatTensor,
) -> Dict[torch.LongTensor, float]:
    #
    model.eval()
    outputs = model.forward(**inputs)
    logits = outputs.logits[torch.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx1, :]
    probs = logits.softmax(dim=-1)

    #
    prob_ratios = {}
    for batch_i in range(prev_probs.size(0)):
        prob_ratios[unmasked_ids1[batch_i]] = [
            mask_idx1[batch_i],
            unmasked_ids2[batch_i, mask_idx2[batch_i]],
            (prev_probs[batch_i] / probs[batch_i][unmasked_ids2[batch_i, mask_idx2[batch_i]]]).item(),
        ]

    return prob_ratios


def filter_bias(male_prob_ratios: Dict[torch.LongTensor, float], female_prob_ratios: Dict[torch.LongTensor, float]):
    #
    filtered_male_ids = {}
    filtered_female_ids = {}

    #
    for k, v in sorted(male_prob_ratios.items(), key=lambda item: item[1], reverse=True):
        if v[-1] > 1.0:
            filtered_male_ids[k] = [v[0], v[1]]

    for k, v in sorted(female_prob_ratios.items(), key=lambda item: item[1], reverse=True):
        if v[-1] > 1.0:
            filtered_female_ids[k] = [v[0], v[1]]

    return filtered_male_ids, filtered_female_ids


def get_switched_sents(
    ids: Dict[torch.LongTensor, torch.LongTensor],
    tokenizer: BertTokenizer,
) -> Dict[str, torch.LongTensor]:
    # preset
    sents = {}

    prev_ids = []
    prev_info = []
    for k, v in ids.items():
        prev_ids.append(k)
        prev_info.append(v)

    prev_ids = torch.stack(prev_ids)

    # from male to female
    new_ids = torch.clone(prev_ids).detach()
    for batch_i in range(new_ids.size(0)):
        new_ids[batch_i][prev_info[batch_i][0]] = prev_info[batch_i][1]
        sents[tokenizer.decode(new_ids[batch_i], skip_special_tokens=True)] = prev_info[batch_i][0]

    return sents


def prepare_masked_inputs_and_labels(sents: Dict[str, torch.LongTensor], tokenizer: BertTokenizer):
    # preset
    input_texts = [k for k in sents.keys()]
    mask_token_idx = [v for v in sents.values()]

    # get unmasked inputs
    inputs = tokenizer(text=input_texts, padding=True, truncation=True, return_tensors="pt")

    # copy unmasked inputs as labels
    labels = torch.clone(inputs["input_ids"]).detach()

    # get masked inputs
    masked_inputs = {}
    for key in inputs.keys():
        masked_inputs[key] = torch.clone(inputs[key]).detach()

    for batch_i in range(torch.Tensor.size(masked_inputs["input_ids"])[0]):
        masked_inputs["input_ids"][batch_i, mask_token_idx[batch_i]] = tokenizer.mask_token_id

    return masked_inputs, labels


def overwrite_state_dict(
    model: DataParallel, model_args: ModelArguments
) -> Tuple[BertForSequenceClassification, BertModel]:
    """Extract and transfer only the trained weights of the layer matching the new model.

    Args:
        trained_model (DebiasRunner): A debiased fine-tuned model.
        model_args (ModelArguments): A parsed model arguments.
    """
    # get initialized pre-trained model
    glue_model = BertForSequenceClassification.from_pretrained(model_args.model_name)
    seat_model = BertModel.from_pretrained(model_args.model_name)

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
