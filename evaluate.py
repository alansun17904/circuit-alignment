import tqdm
import json
import argparse
from typing import List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer

from bbench import *


def pad_left(toks: List[List[int]], pad_token: int):
    """Pad the set of tokens to the same length."""
    mlen = max(len(t) for t in toks)
    return [[pad_token] * (mlen - len(t)) + t for t in toks]


def tokenize_multi_choice(tokenizer, prompts: List[str], choices: List[List[str]]):
    assert len(prompts) == len(
        choices
    ), "There needs to be the same number of questions as answers."
    choice_toks = [tokenizer(cs, add_special_tokens=False, truncation=True).input_ids for cs in choices]
    max_choice_tok_len = max(len(v) for v in choice_toks)
    prompt_toks = tokenizer(prompts, add_special_tokens=False, max_length=tokenizer.model_max_length - max_choice_tok_len, truncation=True).input_ids
    

    prompt_idxs = []
    prompt_choice_toks = []
    for pi, ptoks in enumerate(prompt_toks):
        for ctoks in choice_toks[pi]:
            prompt_choice_toks.append(ptoks + ctoks)
            prompt_idxs.append(pi)
    return prompt_choice_toks


@torch.inference_mode
def evaluate(m: HookedTransformer, dl: DataLoader, run_with_cache: bool = False):
    losses = []
    for tok in tqdm.tqdm(dl):
        losses.append(m(tok, return_type="loss", loss_per_token=True).mean(1))
    return torch.cat(losses)


@torch.no_grad
def post_process(losses, num_choices: int, corr_answer_indices: torch.LongTensor):
    corr_answer_indices = corr_answer_indices.to(losses.device)
    answers = losses.view(losses.shape[0] // num_choices, num_choices)
    return torch.sum(corr_answer_indices == answers.argmin(1)) / len(answers)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model id")
    parser.add_argument("tok_name", type=str, help="pre-trained tokenizer id")
    parser.add_argument("task", help="task to benchmark", choices=TASKS + ["all"])
    parser.add_argument("fname", help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--shots", type=int, help="number of prompting examples", default=0)
    parser.add_argument("--hf_model", default=False, action="store_true", help="huggingface gated model")
    return parser.parse_args()


class MultiChoice(Dataset):
    def __init__(self, toks):
        self.toks = toks

    def __getitem__(self, idx):
        return self.toks[idx]

    def __len__(self):
        return len(self.toks)
    

def eval_task(task, m, atok, bs, shots):
    factuals, _, answers, _ = generate_prompts(task)
    prompts, aidx, choices = generate_choices(task, factuals, answers)
    if shots != 0:
        prompts = make_few_shot_prompt(factuals, answers, shots)
    toked = tokenize_multi_choice(
        atok, prompts, choices
    )
    toked = pad_left(toked, atok.eos_token_id)
    dl = DataLoader(MultiChoice(torch.LongTensor(toked)), batch_size=bs)
    losses = evaluate(m, dl)
    return post_process(losses, len(choices[0]), torch.LongTensor(aidx)), len(choices[0])


if __name__ == "__main__":

    args = parse_args()

    if not args.hf_model:
        atok = AutoTokenizer.from_pretrained(args.tok_name)
        model = HookedTransformer.from_pretrained(args.model_name)
    else:
        atok = AutoTokenizer.from_pretrained(args.tok_name)
        hf_model = AutoModelForCausalLM.from_pretrained(args.tok_name)
        model = HookedTransformer.from_pretrained(args.model_name, hf_model=hf_model, tokenizer=atok)

    atok.truncation_side = "left"

    perf = dict()

    if args.task == "all":
        for t in TASK_CHOICES:
            print("-"*20 + t + "-"*20)
            acc, n_choices = eval_task(t, model, atok, args.batch_size, args.shots)
            perf[t] = {
                "accuracy": acc.item(),
                "chance": 1/n_choices
            }
            
            print("Accuracy", f"{acc.item():.2f}")
            print("Chance", f"{1/n_choices:.2f}")
    else:
        print("-"*20 + args.task + "-"*20)
        acc, n_choices = eval_task(args.task, model, atok, args.batch_size, args.shots)
        print("Accuracy", f"{acc.item():.2f}")
        print("Chance", f"{1/n_choices:.2f}")
        perf[args.task] = {
            "accuracy": acc.item(),
            "chance": 1/n_choices,
        }

    json.dump(perf, open(args.fname, "w"))