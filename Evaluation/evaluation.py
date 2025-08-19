# Install dependencies if needed
# !pip install jsonlines transformers codegen_metrics

import os
import jsonlines
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from codegen_metrics import chrf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# === MODEL TOKEN CONFIG ===
@dataclass
class ModelTokens:
    filename: str
    prefix: str
    suffix: str
    middle: str

    def special_tokens(self):
        return [self.filename, self.prefix, self.suffix, self.middle]

qwen_tokens = ModelTokens("<|file_sep|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
mellum_tokens = ModelTokens("<filename>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>")
codestral_tokens = ModelTokens("+++++", "[PREFIX]", "[SUFFIX]", "[MIDDLE]")

DEFAULT_FILE_SEP = qwen_tokens.filename

# === SAMPLER using transformers ===
class TransformersSampler:
    def __init__(self, model_name: str, special_tokens: list[str], sampling_parameters: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cpu")
        self.special_tokens = special_tokens
        self.sampling_params = sampling_parameters

    def generate(self, prompts):
        completions = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.sampling_params["max_tokens"],
                temperature=self.sampling_params["temperature"],
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            completions.append(full_output[len(prompt):])  # Remove the prompt
        return completions

sampling_params = {
    "temperature": 0.0,
    "max_tokens": 384,
    "seed": 42,
}

# === UTILS ===
def read_dataset(dataset_name: str) -> list[dict]:
    with jsonlines.open(os.path.join("/Users/uswahyusuf/PycharmProjects/ase2025-starter-kit-practice/data", f"{dataset_name}.jsonl"), "r") as f:
        return [datapoint for datapoint in f]

def truncate_context(context: str, prefix: str, suffix: str, max_new_tokens: int, max_tokens: int, encode: Callable) -> str:
    num_tokens = max_new_tokens + len(encode(prefix)) + len(encode(suffix))
    context_lines = context.splitlines(keepends=True)
    truncated_context = ''
    while context_lines:
        curr_line = context_lines.pop(-1)
        line_tokens = len(encode(curr_line))
        num_tokens += line_tokens
        if num_tokens > max_tokens:
            break
        truncated_context = curr_line + truncated_context
    return truncated_context

# === PROMPT BUILDERS ===
def build_prompt(context, filename, prefix, suffix, tokens: ModelTokens, tokenizer, model_name):
    context = context.replace(DEFAULT_FILE_SEP, tokens.filename)
    truncated_context = truncate_context(
        context, prefix, suffix,
        max_new_tokens=sampling_params["max_tokens"], max_tokens=8000,
        encode=tokenizer.tokenize
    )
    parts = [
        f"{truncated_context}\n",
        f"{tokens.filename}{filename}\n",
        f"{tokens.suffix}{suffix}",
        f"{tokens.prefix}{prefix}",
        f"{tokens.middle}"
    ]
    return ''.join(parts)

# === MAIN EXECUTION ===
dataset = read_dataset("python-practice")
answers = read_dataset("answers/answers-python-practice")
contexts = read_dataset("predictions/python-practice-bm25_top_5_chunks_reversed_order")

# === Mellum ===
mellum_tokenizer = AutoTokenizer.from_pretrained("JetBrains/Mellum-4b-sft-python")
mellum_prompts = [
    build_prompt(pred["context"], dp["path"], pred.get("prefix", dp["prefix"]), dp["suffix"],
                 mellum_tokens, mellum_tokenizer, "mellum")
    for dp, pred in zip(dataset, contexts)
]
mellum_sampler = TransformersSampler("JetBrains/Mellum-4b-sft-python", mellum_tokens.special_tokens(), sampling_params)
mellum_completions = mellum_sampler.generate(mellum_prompts)
mellum_chrf = sum([chrf(ans["middle"], pred) for ans, pred in zip(answers, mellum_completions)]) / len(answers)
print("Mellum ChrF:", mellum_chrf)

# === Qwen ===
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
qwen_prompts = [
    build_prompt(pred["context"], dp["path"], pred.get("prefix", dp["prefix"]), dp["suffix"],
                 qwen_tokens, qwen_tokenizer, "qwen")
    for dp, pred in zip(dataset, contexts)
]
qwen_sampler = TransformersSampler("Qwen/Qwen2-7B-Instruct", qwen_tokens.special_tokens(), sampling_params)
qwen_completions = qwen_sampler.generate(qwen_prompts)
qwen_chrf = sum([chrf(ans["middle"], pred) for ans, pred in zip(answers, qwen_completions)]) / len(answers)
print("Qwen ChrF:", qwen_chrf)

# === Codestral ===
codestral_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codestral-2501", trust_remote_code=True)
codestral_prompts = [
    build_prompt(pred["context"], dp["path"], pred.get("prefix", dp["prefix"]), dp["suffix"],
                 codestral_tokens, codestral_tokenizer, "codestral")
    for dp, pred in zip(dataset, contexts)
]
codestral_sampler = TransformersSampler("Salesforce/codestral-2501", codestral_tokens.special_tokens(), sampling_params)
codestral_completions = codestral_sampler.generate(codestral_prompts)
codestral_chrf = sum([chrf(ans["middle"], pred) for ans, pred in zip(answers, codestral_completions)]) / len(answers)
print("Codestral ChrF:", codestral_chrf)
