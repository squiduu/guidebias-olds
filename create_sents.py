import argparse
import json
import re
import unicodedata
import torch
import os
import logging
from typing import List
from tqdm import tqdm
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_logger():
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s", datefmt="%Y-%m-%d %H:%M")
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)

    return logger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--prompt_path", type=str, default="./data/male/male_promts.json")
    parser.add_argument("--output_path", type=str, default="./data/male/male_sents.json")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--gender_type", type=str, default="male")

    return parser.parse_args()


def clean_words(_words: List[str], tokenizer: GPT2Tokenizer):
    words = []
    for _word in _words:
        if tokenizer.convert_tokens_to_ids(_word) != tokenizer.unk_token_id:
            words.append(_word)

    return words


clear_console()
logger = get_logger()
args = get_args()

logger.info("Set CUDA device.")
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

logger.info("Prepare tokenizer and model.")
gpt2_tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"
bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt2_tokenizer.eos_token_id)
model.config.pad_token_id = model.config.eos_token_id
model.cuda().to(device)

logger.info("Load prompts and stereotypes.")
with open(file=args.prompt_path, mode="r") as fp:
    PROMPTS = json.load(fp)
with open(file="./data/stereotype/stereotype_words.json", mode="r") as fp:
    _STEREOTYPES = json.load(fp)
STEREOTYPES = clean_words(_words=_STEREOTYPES, tokenizer=bert_tokenizer)

logger.info(f"Prepare input texts for {args.gender_type}.")
input_texts = []
for i in range(len(PROMPTS)):
    for stereotype in STEREOTYPES:
        input_texts.append(PROMPTS[i] + " " + stereotype)

logger.info("Tokenize input texts for male.")
input_encodings = gpt2_tokenizer(text=input_texts, padding=True, truncation=True, return_tensors="pt")
input_ids = input_encodings["input_ids"]

logger.info(f"Get generated sentences for {args.gender_type}.")
sents = []
for i in tqdm(range(torch.Tensor.size(input_ids)[0] // args.batch_size), desc=f"Get {args.gender_type} sentences"):
    batch_input_ids = input_ids[i * args.batch_size : (i + 1) * args.batch_size]
    batch_input_ids = torch.Tensor.cuda(batch_input_ids, device=device)

    outputs = model.generate(
        input_ids=batch_input_ids,
        max_length=30,
        do_sample=True,
        early_stopping=True,
        num_beams=5,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )

    gen_texts = []
    for j in range(outputs.size(0)):
        gen_text = gpt2_tokenizer.decode(token_ids=outputs[j], skip_special_tokens=True)

        cleaned_text = unicodedata.normalize("NFKD", gen_text)
        cleaned_text = re.sub(r"[^a-zA-Z0-9.'!?()]", " ", cleaned_text)
        cleaned_text = cleaned_text.lower()
        cleaned_text = " ".join(cleaned_text.split())

        # append list without bracket
        gen_texts += cleaned_text

    # append list without bracket
    sents += gen_texts

logger.info(f"Save generated sentences for {args.gender_type}.")
with open(file=args.output_path, mode="w") as fp:
    json.dump(sents, fp)
