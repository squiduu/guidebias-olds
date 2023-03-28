import torch
import json
import re
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
unmasker = BertForMaskedLM.from_pretrained("bert-base-uncased")
unmasker.eval().cuda().to(device)

with open(file=f"./data/male/male_words_new.json", mode="r") as male_fp:
    MALE_WORDS = json.load(male_fp)
with open(file=f"./data/female/female_words_new.json", mode="r") as female_fp:
    FEMALE_WORDS = json.load(female_fp)
with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
    WIKI_WORDS = json.load(wiki_fp)

male_prompts = []
female_prompts = []
with torch.no_grad():
    for word in MALE_WORDS:
        inputs = tokenizer(text=word + "[MASK]", return_tensors="pt")
        mask_idx = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        for key in BatchEncoding.keys(inputs):
            inputs[key] = torch.Tensor.cuda(inputs[key], device=device)
        mask_idx = torch.Tensor.cuda(mask_idx, device=device)

        outputs = unmasker.forward(**inputs)
        probs = outputs.logits[0, mask_idx.item(), :].softmax(dim=-1)

        wiki_probs = {}
        for wiki in WIKI_WORDS:
            if "'s" not in wiki and len(re.findall(r"[^a-zA-Z0-9.'!?()]", wiki)) == 0:
                wiki_probs[wiki] = probs[tokenizer.convert_tokens_to_ids(wiki)].item()

        sorted_wiki_probs = sorted(wiki_probs.items(), key=lambda item: item[1], reverse=True)
        selected_wiki = [i[0] for i in sorted_wiki_probs[:100]]

        for selected_wiki_word in selected_wiki:
            male_prompts.append(word + " " + selected_wiki_word)

with open(file="./data/male/male_promts.json", mode="w") as fp:
    json.dump(male_prompts, fp)

    for word in FEMALE_WORDS:
        inputs = tokenizer(text=word + "[MASK]", return_tensors="pt")
        mask_idx = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        for key in BatchEncoding.keys(inputs):
            inputs[key] = torch.Tensor.cuda(inputs[key], device=device)
        mask_idx = torch.Tensor.cuda(mask_idx, device=device)

        outputs = unmasker.forward(**inputs)
        probs = outputs.logits[0, mask_idx.item(), :].softmax(dim=-1)

        wiki_probs = {}
        for wiki in WIKI_WORDS:
            if "'s" not in wiki and len(re.findall(r"[^a-zA-Z0-9.'!?()]", wiki)) == 0:
                wiki_probs[wiki] = probs[tokenizer.convert_tokens_to_ids(wiki)].item()

        sorted_wiki_probs = sorted(wiki_probs.items(), key=lambda item: item[1], reverse=True)
        selected_wiki = [i[0] for i in sorted_wiki_probs[:100]]

        for selected_wiki_word in selected_wiki:
            female_prompts.append(word + " " + selected_wiki_word)

with open(file="./data/female/female_promts.json", mode="w") as fp:
    json.dump(female_prompts, fp)

print("Done.")
