import json
import random
from logging import Logger

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import wandb
from config import DataArguments, ModelArguments, TrainingArguments
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from utilities_v10 import (
    clean_words,
    clear_console,
    filter_wiki,
    get_logger,
    prepare_masked_sents,
    prepare_models_and_tokenizer,
)


def run_guidebias(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    """Generate augmented data with stereotype words and save it using the [MASK] token bias.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info(f"Set device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_runner = wandb.init(project=train_args.project, entity="squiduu", name=train_args.run_name)

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare models and tokenizer: {model_args.model_name}")
    fixed_model, tuning_model, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
    fixed_model.to(device)
    tuning_model.to(device)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=tuning_model.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    logger.info("Prepare gender words.")
    with open(file=f"./data/male/male_words_{data_args.num_target_words}.json", mode="r") as male_fp:
        M_WORDS = json.load(male_fp)
    M_WORDS = M_WORDS[: data_args.num_target_words]
    M_WORDS = clean_words(words=M_WORDS, tokenizer=tokenizer)
    M_IDS = tokenizer.convert_tokens_to_ids(M_WORDS)
    with open(file=f"./data/female/female_words_{data_args.num_target_words}.json", mode="r") as female_fp:
        F_WORDS = json.load(female_fp)
    F_WORDS = F_WORDS[: data_args.num_target_words]
    F_WORDS = clean_words(words=F_WORDS, tokenizer=tokenizer)
    F_IDS = tokenizer.convert_tokens_to_ids(F_WORDS)

    NON_GENDER_IDS = [i for i in range(tokenizer.vocab_size) if i not in (M_IDS + F_IDS)]

    logger.info("Prepare stereotype words.")
    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREO_WORDS = json.load(ster_fp)

    logger.info("Prepare wiki words.")
    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        W_WORDS = json.load(wiki_fp)
    W_WORDS = filter_wiki(wiki_words=W_WORDS, gender_words=M_WORDS + F_WORDS, stereo_words=STEREO_WORDS)
    W_WORDS = W_WORDS[: data_args.num_wiki_words]

    #
    sents = prepare_masked_sents(tokenizer=tokenizer, wiki_words=W_WORDS, stereo_words=STEREO_WORDS)

    #
    dl = DataLoader(
        dataset=[i for i in range(len(sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    #
    num_training_steps = int(train_args.num_epochs * len(dl))
    num_warmup_steps = int(num_training_steps * train_args.warmup_proportion)
    logger.info(f"Set lr scheduler with {num_warmup_steps} warm-up steps.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    #
    for ep in range(1, int(train_args.num_epochs) + 1):
        #
        optimizer.zero_grad()

        #
        dl = tqdm(dl)
        for iter, batch_idx in enumerate(dl):
            #
            sents_batch = [sents[torch.Tensor.item(i)] for i in batch_idx]

            #
            inputs = tokenizer(text=sents_batch, padding=True, truncation=True, return_tensors="pt")
            #
            for key in inputs.keys():
                inputs[key] = torch.Tensor.cuda(inputs[key], device=device)

            #
            with amp.autocast_mode.autocast():
                with torch.no_grad():
                    fixed_outputs = fixed_model.forward(**inputs)
                tuning_outputs = tuning_model.forward(**inputs)

                #
                fixed_logits = fixed_outputs.logits[:, 1, :].softmax(dim=-1)
                tuning_logits = tuning_outputs.logits[:, 1, :].softmax(dim=-1)

                #
                bias_prob = []
                for i in range(len(M_IDS)):
                    bias_prob.append((tuning_logits[:, M_IDS[i]] - tuning_logits[:, F_IDS[i]]).abs())
                bias_prob = torch.stack(bias_prob).mean()
                #
                lm_prob = (fixed_logits[:, NON_GENDER_IDS] - tuning_logits[:, NON_GENDER_IDS]).abs().mean()

                #
                m_sents = []
                f_sents = []
                for i in range(len(sents_batch)):
                    for j in range(len(M_WORDS)):
                        m_sents.append(str.replace(sents_batch[i], "[MASK]", M_WORDS[j]))
                        f_sents.append(str.replace(sents_batch[i], "[MASK]", F_WORDS[j]))

                #
                stereo_m_lens = []
                stereo_f_lens = []
                for i in range(len(m_sents)):
                    stereo_m_lens.append(len(tokenizer.tokenize(str.split(m_sents[i])[2])))
                    stereo_f_lens.append(len(tokenizer.tokenize(str.split(f_sents[i])[2])))
                stereo_m_lens = torch.tensor(stereo_m_lens)
                stereo_f_lens = torch.tensor(stereo_f_lens)

                #
                m_inputs = tokenizer(text=m_sents, padding=True, truncation=True, return_tensors="pt")
                f_inputs = tokenizer(text=m_sents, padding=True, truncation=True, return_tensors="pt")

                for key in m_inputs.keys():
                    m_inputs[key] = torch.Tensor.cuda(m_inputs[key], device=device)
                    f_inputs[key] = torch.Tensor.cuda(f_inputs[key], device=device)

                #
                tuning_m_outputs = tuning_model.forward(**m_inputs, output_hidden_states=True)
                tuning_f_outputs = tuning_model.forward(**f_inputs, output_hidden_states=True)

                #
                cossims = []
                for i in range(tuning_m_outputs.hidden_states[-1].size(0)):
                    m_cossim = F.cosine_similarity(
                        tuning_m_outputs.hidden_states[-1][i, 1, :],
                        tuning_m_outputs.hidden_states[-1][i, torch.arange(3, 3 + stereo_m_lens[i]), :].mean(dim=0),
                        dim=-1,
                    )
                    f_cossim = F.cosine_similarity(
                        tuning_f_outputs.hidden_states[-1][i, 1, :],
                        tuning_f_outputs.hidden_states[-1][i, torch.arange(3, 3 + stereo_f_lens[i]), :].mean(dim=0),
                        dim=-1,
                    )
                    cossims.append(m_cossim - f_cossim)

                bias_cossim = torch.stack(cossims).mean()

                #
                new_sents = []
                for i in range(len(m_sents)):
                    m_temp = str.replace(m_sents[i], str.split(m_sents[i])[2], random.choice(W_WORDS))
                    f_temp = str.replace(f_sents[i], str.split(f_sents[i])[2], random.choice(W_WORDS))
                    new_sents.append(" ".join(m_temp))
                    new_sents.append(" ".join(f_temp))

                #
                new_inputs = tokenizer(text=new_sents, padding=True, truncation=True, return_tensors="pt")
                for key in new_inputs.keys():
                    new_inputs[key] = torch.Tensor.cuda(new_inputs[key], device=device)

                with torch.no_grad():
                    fixed_new_outputs = fixed_model.forward(**new_inputs, output_hidden_states=True)
                tuning_new_outputs = tuning_model.forward(**new_inputs, output_hidden_states=True)

                #
                fixed_new_hidden = fixed_new_outputs.hidden_states[-1].mean(dim=1)
                tuning_new_hidden = tuning_new_outputs.hidden_states[-1].mean(dim=1)

                #
                lm_kld = F.kl_div(
                    F.log_softmax(tuning_new_hidden, dim=-1), F.softmax(fixed_new_hidden, dim=-1), reduction="batchmean"
                )
                lm_cossim = 1 - F.cosine_similarity(tuning_new_hidden, fixed_new_hidden).mean()

                #
                loss = bias_prob + lm_prob + bias_cossim + lm_kld + lm_cossim

            #
            scaler.scale(loss / train_args.grad_accum_steps).backward()

            #
            scale = scaler.get_scale()
            skip_scheduler = scale != scaler.get_scale()
            if not skip_scheduler:
                scheduler.step()

            #
            if (iter + 1) % train_args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            #
            dl.set_description(
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Lr: {optimizer.param_groups[0]['lr']:.9f} - Loss: {loss:.4f} - Bias Prob: {bias_prob:.4f} - LM Prob: {lm_prob:.4f} - Bias Cossim: {bias_cossim:.4f} - LM KLD: {lm_kld:.4f} - LM Cosssim: {lm_cossim:.4f}"
            )
            wandb_runner.log(
                {
                    "Total Loss": loss,
                    "Bias Prob": bias_prob,
                    "LM Prob": lm_prob,
                    "Bias Cossim": bias_cossim,
                    "LM KLD": lm_kld,
                    "LM Cossim": lm_cossim,
                }
            )

        # after training
        logger.info("Save a fine-tuned model.")
        tuning_model.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")
        tokenizer.save_pretrained(f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    run_guidebias(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
