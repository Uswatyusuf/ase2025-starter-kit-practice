from dataclasses import dataclass
from typing import List, Tuple
import csv
import matplotlib.pyplot as plt
import tiktoken
import random
import argparse
import os, jsonlines, concurrent.futures, re
from tqdm import tqdm
import numpy as np
import torch
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin
from tree_sitter import Language, Parser

from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaLLM
from chunking import basic_ast_chunk_code, basic_ast_chunk_code_methods_only, chunk_kotlin_with_full_context, \
    basic_ast_chunk_code_methods_only_with_desc
from transformers import AutoTokenizer, AutoModel

PY_LANGUAGE = Language(tspython.language())
KT_LANGUAGE = Language (ts_kotlin.language())

argparser = argparse.ArgumentParser()
# Parameters for context collection strategy
argparser.add_argument("--stage", type=str, default="practice", help="Stage of the project")
argparser.add_argument("--lang", type=str, default="python", help="Language")
argparser.add_argument("--strategy", type=str, default="random", help="Context collection strategy")

# Parameters for context trimming
argparser.add_argument("--trim-prefix", action="store_true", help="Trim the prefix to 10 lines")
argparser.add_argument("--trim-suffix", action="store_true", help="Trim the suffix to 10 lines")

args = argparser.parse_args()

stage = args.stage
language = args.lang
strategy = args.strategy

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"Running the {strategy} baseline for stage '{stage}'")

# token used to separate different files in the context
FILE_SEP_SYMBOL = "<|file_sep|>"
# format to compose context from a file
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

# Adds ollama for generating textual info

SYSTEM_INSTRUCTION_MID = """You are a professional developer.

Below, you are given two partial code fragments: a PREFIX and a SUFFIX from the same source file. Together, these represent parts of a larger file.

Please write a high-level description that subtly hints at what the missing middle code is likely doing.

Focus on **specific actions** the code performs, such as initializing connections, creating data structures, or invoking key functions. Include any **key purposes or behaviors**, such as validating workflows or testing performance characteristics.

Avoid using quotes, colons, or meta-language like 'Hint:'.

Start your sentence with "The part requiring completion" followed by its **likely actions and objectives**.

Keep your description clear and concise (1-2 sentences).
"""

SYSTEM_INSTRUCTION_W_TARGET_FILE = """You are an experienced software engineer.

Below, you are given two partial code fragments: a PREFIX and a SUFFIX from the same source file. Together, these represent parts of a larger file.

Please write a brief, high-level description of the **purpose** of this file, based on the given code. Focus on describing what the file is supposed to do overall (its main functionality or role in the project).

Keep your description short and clear (ideally 1-3 sentences).

"""

SYSTEM_INSTRUCTION_PREFIX = """You are an experienced software engineer.

You are given the PREFIX of a source file. This represents the first part of a larger file, with the rest missing.

Please provide a brief, high-level summary of what this segment appears to do. Focus on the purpose or setup it establishes for the rest of the file.

Keep your description clear and concise in just a sentence.

Start your sentence with "This segment" followed by its function/ use.
"""


SYSTEM_INSTRUCTION_SUFFIX = """You are an experienced software engineer.

You are given the SUFFIX of a source file. This represents the ending portion of a larger file, with earlier parts missing.

Write a brief, high-level summary of what this segment appears to complete or finalize. Focus on the file’s final behavior, result handling, or cleanup logic.

Keep your description clear and concise in just a sentence.

Start your sentence with "This segment" followed by its function/ use.
"""

SYSTEM_INSTRUCTION_BM25_QUERY = """You are a senior software developer completing a missing block of code.

You're provided with a PREFIX and SUFFIX from a file. Your job is to analyze the two and write a **detailed, technically rich description** of what the missing middle code is supposed to do.

Focus on:
- The main task or goal of the middle section.
- Specific methods, libraries, or components that are likely used.
- Expected control flow (e.g., loops, conditionals).
- Data passed between the prefix and suffix.
- Any system-specific behavior or constraints (e.g., retries, serialization, priorities).

Avoid:
- Lists, bullet points, headings
- Code formatting or syntax
- Expected code structure
- Example Implementation Logic
- Overly abstract/general language like “some logic” or “handles behavior”

Make the description clear and structured, like a brief implementation plan or comment block — this will be used as a **search query** to retrieve related code.

Be precise and informative.
"""

SYSTEM_INSTRUCTION_CODE_GEN = """
You are a professional software developer.

You are given:
- A PREFIX and a SUFFIX from the same source file.
- CONTEXT from other relevant files in the same repository.

Your task: 
- Generate the missing middle code between the PREFIX and SUFFIX (Fill in the middle logic).
- Use the CONTEXT to ensure consistency, correctness, and completeness.
- Your response should only be the missing code.
"""


llm = OllamaLLM(model="qwen3:8b", temperature=0)

llm_code_completion = OllamaLLM(model="qwen2.5-coder:14b")

mellum_tokenizer = AutoTokenizer.from_pretrained("JetBrains/Mellum-4b-sft-python")


# Load UniXcoder model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")
model.eval()  # disable dropout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prepare_bm25_str(s: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in s.lower()).split()

def count_tokens(text: str) -> int:
    # Use tokenizer.encode instead of .tokenize for accuracy
    return len(mellum_tokenizer.encode(text, add_special_tokens=False))

def count_tokens_unix(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_chunks_for_file(file_path: str, chunk_cache: dict, debug_dir: str = None) -> list[tuple[str, str]]:
    if file_path in chunk_cache:
        return chunk_cache[file_path]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    if language == "python":
        chunks = basic_ast_chunk_code(content)
    elif language == "kotlin":
        chunks = chunk_kotlin_with_full_context(content)
    chunk_entries = [(file_path, chunk) for chunk in chunks]
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, os.path.basename(file_path) + ".chunks.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            for i, (_, chunk) in enumerate(chunk_entries):
                f.write(f"--- Chunk {i+1} ---\n{chunk.strip()}\n\n")
    chunk_cache[file_path] = chunk_entries
    return chunk_entries

def assemble_context(top_chunks, file_descriptions, root_dir):
    context_parts = []
    for file_path, chunk_content in top_chunks:
        clean_file_name = file_path[len(root_dir) + 1:]
        file_desc = re.sub(r'<think>.*?</think>', '', file_descriptions[file_path], flags=re.DOTALL).strip()

        context_part = (
            f"{FILE_SEP_SYMBOL}{clean_file_name}\n"
            f"# This code snippet comes from this file: {clean_file_name},{file_desc}\n\n "
            f"{chunk_content.strip()}\n"
        )
        context_parts.append(context_part)
    return context_parts

def trim_rest_code_with_tokenizer(rest_of_code: str, import_tokens: int, max_tokens: int) -> str:
    rest_lines = rest_of_code.splitlines()
    trimmed_lines = []
    total_tokens = import_tokens

    for line in reversed(rest_lines):
        line_token_count = count_tokens_unix(line)
        if total_tokens + line_token_count > max_tokens:
            break
        trimmed_lines.insert(0, line)
        total_tokens += line_token_count

    return '\n'.join(trimmed_lines)


def find_random_file(root_dir: str, min_lines: int = 10) -> str:
    """
    Select a random file:
        - in the given language
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files with given extension.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected random file or None if no files were found.
    """
    code_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            code_files.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass
    return random.choice(code_files) if code_files else None


def find_bm25_file(root_dir: str, prefix: str, suffix: str, min_lines: int = 10) -> str:
    """
    Select the file:
        - in the given language
        - with the highest BM25 score with the completion file
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param prefix: Prefix of the completion file.
    :param suffix: Suffix of the completion file.
    :param min_lines: Minimum number of lines required in the file.
    :return:
    """

    corpus = []
    file_names = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            content = prepare_bm25_str(content)
                            corpus.append(content)
                            file_names.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    query = (prefix + " " + suffix).lower()
    query = prepare_bm25_str(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    best_idx = scores.argmax()

    return file_names[best_idx] if file_names else None


def describe_file(file_path: str) -> str:
    """
    Uses the LLM to generate a description for what the whole file does.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    file_desc_prompt = f"""
    You are a senior software engineer writing a brief code summary for documentation.

    TASK:
    Summarize the content of the file below in EXACTLY 1 to 3 sentences.

    RULES:
    - Be direct and factual.
    - Cover key responsibilities, important functions/classes, and main behaviors.
    - Avoid generic phrases or commentary.
    - Any response longer than 3 sentences will be discarded.

    EXAMPLE:
    If the file content is:
    --- BEGIN FILE ---
    def add(a, b):
        return a + b
    --- END FILE ---

    The summary should be:
    "This file defines a simple addition function that takes two numbers and returns their sum."

    Now, summarize the following file:

    --- BEGIN FILE ---
    <file_content>
    {content}
    <file_content>
    --- END FILE ---

    Provide ONLY the summary (1-3 sentences), nothing else.
    """

    try:
        return llm.invoke(file_desc_prompt).strip()
    except Exception as e:
        print(f"Failed to describe {file_path}: {e}")
        return "[No description available]"

def describe_files_for_top_chunks(top_chunks, description_cache):
    unique_files = set(file_path for file_path, _ in top_chunks)
    file_descriptions = {}

    for file_path in unique_files:
        if file_path in description_cache:
            file_descriptions[file_path] = description_cache[file_path]
        else:
            desc = describe_file(file_path)
            description_cache[file_path] = desc
            file_descriptions[file_path] = desc

    return file_descriptions


def find_bm25_top_3_files(root_dir: str, prefix: str, suffix: str, min_lines: int = 10, no_of_files: int = 4) -> List[str]:
    """
    Select the top three files:
        - in the given language
        - with the highest BM25 score with the completion file
        - in the given directory and its subdirectories
        - meeting length requirements

    :param no_of_files:
    :param root_dir: Directory to search for files.
    :param prefix: Prefix of the completion file.
    :param suffix: Suffix of the completion file.
    :param min_lines: Minimum number of lines required in the file.
    :return:
    """

    corpus = []
    file_names = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            content = prepare_bm25_str(content)
                            corpus.append(content)
                            file_names.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    query = (prefix + " " + suffix).lower()
    query = prepare_bm25_str(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:no_of_files]

    # Reverse the order to go from lowest of top 3 to highest
    #reversed_top_indices = top_indices[::-1]

    return [file_names[i] for i in top_indices]


def bm25_top_n_chunks(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 5) -> list[tuple[str, str]]:
    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache, f"debug-dir-methods-only_{language}"))

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(prefix + " " + suffix)

    if not chunk_texts:
        return []

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    reversed_top_indices = top_indices[::-1]

    return [all_chunks[i] for i in reversed_top_indices]

def bm25_top_n_chunks(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 5) -> list[tuple[str, str]]:
    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache, f"debug-dir-methods-only_{language}"))

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(prefix + " " + suffix)

    if not chunk_texts:
        return []

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    reversed_top_indices = top_indices[::-1]

    return [all_chunks[i] for i in reversed_top_indices]

def bm25_top_n_chunks_with_tokens(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 5) -> List[Tuple[str, str, int]]:
    all_chunks = []
    chunk_cache = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                chunks = get_chunks_for_file(file_path, chunk_cache)
                all_chunks.extend(chunks)

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(prefix + " " + suffix)

    if not chunk_texts:
        return []

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    reversed_top_indices = top_indices[::-1]
    result = []
    for idx in reversed_top_indices:
        file_path, chunk = all_chunks[idx]
        token_count = count_tokens(chunk)
        result.append((file_path, chunk, token_count))

    return result

def bm25_top_n_chunks_with_missing_code_description(
    root_dir: str,
    bm25_middle_description: str,
    ext: str,
    top_k: int = 5
) -> list[tuple[str, str]]:
    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache))

    if not all_chunks:
        return []

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    print(bm25_middle_description)
    query = prepare_bm25_str(bm25_middle_description)

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    reversed_top_indices = top_indices[::-1]
    return [all_chunks[i] for i in reversed_top_indices]


def bm25_chunks_within_limit_sorted_low_to_high(
    root_dir: str,
    prefix: str,
    suffix: str,
    ext: str,
    context_limit: int = 8000
) -> tuple[list[tuple[str, str]], int]:
    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache))

    if not all_chunks:
        return [], 0

    query = prepare_bm25_str(prefix + " " + suffix)
    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected_chunks = []
    context_parts = []

    prefix_tokens = count_tokens(prefix)
    suffix_tokens = count_tokens(suffix)
    total_tokens = prefix_tokens + suffix_tokens

    for i in ranked_indices:
        file_path, chunk_content = all_chunks[i]
        clean_file_name = file_path[len(root_dir) + 1:]
        context_part = FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL,
            file_name=clean_file_name,
            file_content=chunk_content
        )

        chunk_tokens = count_tokens(context_part)
        if total_tokens + chunk_tokens > context_limit:
            break
        selected_chunks.append((file_path, chunk_content))
        context_parts.append(context_part)
        total_tokens += chunk_tokens

    # Sort final selected chunks from low to high BM25 (i.e., weak to strong match)
    selected_chunks = sorted(
        selected_chunks,
        key=lambda x: scores[all_chunks.index(x)]
    )
    return selected_chunks, total_tokens

def bm25_chunks_above_percentile(
    root_dir: str,
    prefix: str,
    suffix: str,
    ext: str,
    percentile: float = 90.0,
    histogram_dir: str = None,
    context_limit: int = 8000,
) -> list[tuple[str, str]]:
    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache))

    if not all_chunks:
        return []

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(prefix + " " + suffix)
    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)

    threshold = np.percentile(scores, percentile)
    selected = [(chunk, score) for chunk, score in zip(all_chunks, scores) if score >= threshold]

    # Plot histogram
    if histogram_dir:
        os.makedirs(histogram_dir, exist_ok=True)
        repo_name = os.path.basename(root_dir)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(scores, bins=50, color='skyblue', edgecolor='black')
        ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2,
                   label=f"{percentile}th percentile = {threshold:.2f}")
        ax.set_title(f"BM25 Score Distribution for {repo_name}")
        ax.set_xlabel("BM25 Score")
        ax.set_ylabel("Number of Chunks")
        ax.legend()
        save_path = os.path.join(histogram_dir, f"{repo_name}_bm25_hist.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    enc = tiktoken.encoding_for_model("gpt-4")
    total_tokens = len(enc.encode(prefix)) + len(enc.encode(suffix))

    # Sort by BM25 score high → low for greedy selection
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)

    included_chunks = []
    for (file_path, chunk_content), _ in selected_sorted:
        context_part = FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL,
            file_name=file_path[len(root_dir) + 1:],
            file_content=chunk_content
        )
        chunk_tokens = len(enc.encode(context_part))
        if total_tokens + chunk_tokens > context_limit:
            break
        included_chunks.append((file_path, chunk_content))
        total_tokens += chunk_tokens

    return sorted(included_chunks, key=lambda x: scores[all_chunks.index(x)], reverse=True)

def find_random_recent_file(root_dir: str, recent_filenames: list[str], min_lines: int = 10) -> str:
    """
    Select the most recent file:
        - in the given language
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param recent_filenames: List of recent files filenames.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected random file or None if no files were found.
    """
    code_files = []
    for filename in recent_filenames:
        if filename.endswith(extension):
            file_path = os.path.join(root_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= min_lines:
                        code_files.append(file_path)
            except Exception as e:
                # Optional: handle unreadable files
                # print(f"Could not read {file_path}: {e}")
                pass
    return random.choice(code_files) if code_files else None

def trim_prefix(prefix: str):
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > 10:
        prefix = "\n".join(prefix_lines[-10:])
    return prefix

def trim_suffix(suffix: str):
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > 10:
        suffix = "\n".join(suffix_lines[:10])
    return suffix

def log_token_usage(token_log_path: str, instance_id: int, files: List[str], token_count: int, prefix_suffix_tokens: int):
    file_list = ", ".join(files)
    file_exists = os.path.isfile(token_log_path)
    with open(token_log_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Instance', 'Files', 'PrefixSuffixTokens', 'TotalTokens']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Instance': instance_id,
            'Files': file_list,
            'PrefixSuffixTokens': prefix_suffix_tokens,
            'TotalTokens': token_count
        })

def plot_token_usage_chart(token_log_path: str):
    instances = []
    total_tokens = []
    prefix_suffix_tokens = []

    with open(token_log_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            instances.append(int(row['Instance']))
            total = int(row['TotalTokens'])
            prefix_suffix = int(row['PrefixSuffixTokens'])
            total_tokens.append(total)
            prefix_suffix_tokens.append(prefix_suffix)

    # Compute chunk tokens (i.e., total - prefix+suffix)
    chunk_tokens = [total - prefix for total, prefix in zip(total_tokens, prefix_suffix_tokens)]

    plt.figure(figsize=(14, 6))
    bar1 = plt.bar(instances, prefix_suffix_tokens, color='#6baed6', label='Prefix + Suffix')
    bar2 = plt.bar(instances, chunk_tokens, bottom=prefix_suffix_tokens, color='orange', label='Context Chunks')

    plt.axhline(8000, color='orange', linestyle='--', label='8K Token Limit')
    plt.axhline(16000, color='red', linestyle='--', label='16K Token Limit')
    plt.xlabel("Code Completion Instance")
    plt.ylabel("Total Token Count")
    plt.title("Token Length per Code Completion Context")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions/token_chart.png")
    plt.show()



# Define the log path
token_log_path = os.path.join("predictions", f"{language}-{stage}-{strategy}_token_usage.csv")

# Path to the file with completion points
completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")

# Path to the file to store predictions
prediction_file_name = f"{language}-{stage}-{strategy}"
if args.trim_prefix:
    prediction_file_name += "-short-prefix"
if args.trim_suffix:
    prediction_file_name += "-short-suffix"
predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

instance_id = 0
descriptions_log_path = os.path.join("predictions", f"{language}-{stage}-{strategy}_descriptions.jsonl")
with jsonlines.open(completion_points_file, 'r') as reader:
    with jsonlines.open(predictions_file, 'w') as writer:
        for instance_id, datapoint in enumerate(reader):
            repo_id = datapoint['id']
            repo_path = datapoint['repo'].replace("/", "__")
            repo_revision = datapoint['revision']
            root_directory = os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{repo_revision}")
            prefix = datapoint["prefix"]
            suffix = datapoint["suffix"]
            selected_files = []
            # Compose the full context with file separators
            used_tokens = 0
            context_parts = []

            # Run the strategy
            if strategy == "random":
                selected_files = [find_random_file(root_directory)]
            elif strategy == "bm25":
                selected_files = [find_bm25_file(root_directory, datapoint['prefix'], datapoint['suffix'])]
            elif strategy == "bm25_top_4_files":
                selected_files = find_bm25_top_3_files(root_directory, datapoint['prefix'], datapoint['suffix'])
            elif strategy == "bm25_top_5_chunks_methods_only_with_desc":
                top_chunks= bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
                output_file = f"debug-chunk-methods/top_chunks_{language}_{repo_id}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                        f.write(f"# Root Directory: {root_directory}\n\n")
                        for file_path, chunk_content in top_chunks:
                            clean_file_name = file_path[len(root_directory) + 1:]
                            header = f"## File: {clean_file_name}\n"
                            f.write(header)
                            f.write(chunk_content)
                            f.write("\n\n")  # Separate chunks clearly
            elif strategy == "bm25_top_5_chunks_methods_only_with_desc":
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
                output_file = f"debug-chunk-methods/top_chunks_{language}_{repo_id}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"# Root Directory: {root_directory}\n\n")
                    for file_path, chunk_content in top_chunks:
                        clean_file_name = file_path[len(root_directory) + 1:]
                        header = f"## File: {clean_file_name}\n"
                        f.write(header)
                        f.write(chunk_content)
                        f.write("\n\n")  # Separate chunks clearly
            elif strategy == "bm25_chunks_text_info_mid_hint_no_other_context":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_MID}

                --- BEGIN PREFIX ---
                {original_prefix}
                --- END PREFIX ---

                --- BEGIN SUFFIX ---
                {original_suffix}
                --- END SUFFIX ---

                """

                try:
                    middle_description = llm.invoke(fill_prompt).strip()
                    # Clean up stray outer quotes if present
                    if middle_description.startswith('"') and middle_description.endswith('"'):
                        middle_description = middle_description[1:-1]

                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    middle_description = "[LLM Description Unavailable]"
                middle_description = re.sub(r'<think>.*?</think>', '', middle_description, flags=re.DOTALL).strip()
                description_record = {
                    "instance_id": instance_id,
                    "repo": datapoint["repo"],
                    "description": middle_description
                }
                with jsonlines.open(descriptions_log_path, 'a') as descriptions_writer:
                    descriptions_writer.write(description_record)
                description_comment = "\n".join(f"# {line}" for line in middle_description.splitlines())

                # STEP 3: Prepend to prefix
                prefix = f"Hint: The missing part might; {description_comment}\n\n{original_prefix}"
                suffix = original_suffix

                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = ""
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_mid_desc_bm25_query":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_BM25_QUERY}

                    --- BEGIN PREFIX ---
                    {original_prefix}
                    --- END PREFIX ---

                    --- BEGIN SUFFIX ---
                    {original_suffix}
                    --- END SUFFIX ---

                    """

                try:
                    bm25_middle_description = llm.invoke(fill_prompt).strip()
                    # Clean up stray outer quotes if present
                    if bm25_middle_description.startswith('"') and bm25_middle_description.endswith('"'):
                        bm25_middle_description = bm25_middle_description[1:-1]

                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    bm25_middle_description = "[LLM Description Unavailable]"
                bm25_middle_description = re.sub(r'<think>.*?</think>', '', bm25_middle_description, flags=re.DOTALL).strip()
                top_chunks = bm25_top_n_chunks_with_missing_code_description(root_directory, bm25_middle_description, extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_top_5_chunks_target_file_text_info":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_W_TARGET_FILE}

                                --- BEGIN PREFIX ---
                                {original_prefix}
                                --- END PREFIX ---

                                --- BEGIN SUFFIX ---
                                {original_suffix}
                                --- END SUFFIX ---

                                """

                try:
                    file_description = llm.invoke(fill_prompt).strip()
                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    file_description = "[LLM Description Unavailable]"
                file_description = re.sub(r'<think>.*?</think>', '', file_description, flags=re.DOTALL).strip()
                description_record = {
                    "instance_id": instance_id,
                    "repo": datapoint["repo"],
                    "description": file_description
                }
                with jsonlines.open(descriptions_log_path, 'a') as descriptions_writer:
                    descriptions_writer.write(description_record)
                description_comment = "\n".join(f"# {line}" for line in file_description.splitlines())

                # STEP 3: Prepend to prefix
                prefix = f"{description_comment}\n\n{original_prefix}"
                suffix = original_suffix
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                           file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_text_info_hint":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_MID}

                --- BEGIN PREFIX ---
                {original_prefix}
                --- END PREFIX ---

                --- BEGIN SUFFIX ---
                {original_suffix}
                --- END SUFFIX ---

                """

                try:
                    middle_description = llm.invoke(fill_prompt).strip()
                    # Clean up stray outer quotes if present
                    if middle_description.startswith('"') and middle_description.endswith('"'):
                        middle_description = middle_description[1:-1]

                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    middle_description = "[LLM Description Unavailable]"
                middle_description = re.sub(r'<think>.*?</think>', '', middle_description, flags=re.DOTALL).strip()
                description_record = {
                    "instance_id": instance_id,
                    "repo": datapoint["repo"],
                    "description": middle_description
                }
                with jsonlines.open(descriptions_log_path, 'a') as descriptions_writer:
                    descriptions_writer.write(description_record)
                description_comment = "\n".join(f"# {line}" for line in middle_description.splitlines())

                # STEP 3: Prepend to prefix
                prefix = f"{description_comment}\n\n{original_prefix}"
                suffix = original_suffix

                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_target_file_and_mis_code_text_info":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                target_file_prompt = f"""{SYSTEM_INSTRUCTION_W_TARGET_FILE}

                                --- BEGIN PREFIX ---
                                {original_prefix}
                                --- END PREFIX ---

                                --- BEGIN SUFFIX ---
                                {original_suffix}
                                --- END SUFFIX ---

                                """
                mis_code_prompt = f"""{SYSTEM_INSTRUCTION_MID}

                                --- BEGIN PREFIX ---
                                {original_prefix}
                                --- END PREFIX ---

                                --- BEGIN SUFFIX ---
                                {original_suffix}
                                --- END SUFFIX ---

                                """
                try:
                    file_description = llm.invoke(target_file_prompt).strip()
                    mis_code_description = llm.invoke(mis_code_prompt).strip()
                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    file_description = "[LLM Description Unavailable]"
                    mis_code_description = "[LLM Description Unavailable]"
                file_description = re.sub(r'<think>.*?</think>', '', file_description, flags=re.DOTALL).strip()
                mis_code_description = re.sub(r'<think>.*?</think>', '', mis_code_description, flags=re.DOTALL).strip()
                description_record = {
                    "instance_id": instance_id,
                    "repo": datapoint["repo"],
                    "file_description": file_description,
                    "mis_code_description": mis_code_description
                }
                with jsonlines.open(descriptions_log_path, 'a') as descriptions_writer:
                    descriptions_writer.write(description_record)
                file_description_comment = "\n".join(f"# {line}" for line in file_description.splitlines())
                mis_code_description_comment = "\n".join(f"# {line}" for line in mis_code_description.splitlines())
                # STEP 3: Prepend to prefix
                prefix = f"{file_description_comment}\n\n{mis_code_description_comment}\n\n{original_prefix}"
                suffix = f"{original_suffix}"
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,

                                                           file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_text_info_px_sx":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_MID}

                --- BEGIN PREFIX ---
                {original_prefix}
                --- END PREFIX ---

                --- BEGIN SUFFIX ---
                {original_suffix}
                --- END SUFFIX ---

                """

                # STEP 2: Query LLM for a description of PREFIX
                prefix_prompt = f"""{SYSTEM_INSTRUCTION_PREFIX}

                                --- BEGIN PREFIX ---
                                {original_prefix}
                                --- END PREFIX ---
                                """

                # STEP 2: Query LLM for a description of PREFIX
                suffix_prompt = f"""{SYSTEM_INSTRUCTION_SUFFIX}

                                                --- BEGIN PREFIX ---
                                                {original_suffix}
                                                --- END PREFIX ---
                                                """

                try:
                    prefix_description = llm.invoke(prefix_prompt).strip()
                    #middle_description = llm.invoke(fill_prompt).strip()
                    suffix_description = llm.invoke(suffix_prompt).strip()


                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    middle_description = "[LLM Description Unavailable]"
                    prefix_description = "[LLM Description Unavailable]"
                    suffix_description = "[LLM Description Unavailable]"
                prefix_description = re.sub(r'<think>.*?</think>', '', prefix_description, flags=re.DOTALL).strip()
                #middle_description = re.sub(r'<think>.*?</think>', '', middle_description, flags=re.DOTALL).strip()
                suffix_description = re.sub(r'<think>.*?</think>', '', suffix_description, flags=re.DOTALL).strip()
                description_record = {
                    "instance_id": instance_id,
                    "repo": datapoint["repo"],
                    "px_description": prefix_description,
                    "sx_description": suffix_description
                }
                with jsonlines.open(descriptions_log_path, 'a') as descriptions_writer:
                    descriptions_writer.write(description_record)
                px_description_comment = "\n".join(f"# {line}" for line in prefix_description.splitlines())
                #mid_description_comment = "\n".join(f"# {line}" for line in middle_description.splitlines())
                sx_description_comment = "\n".join(f"# {line}" for line in suffix_description.splitlines())

                # STEP 3: Prepend to prefix
                prefix = f"{px_description_comment}\n{original_prefix}"
                suffix = f"{sx_description_comment}\n{original_suffix}"

                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_limited_8k":
                    description_cache = {}
                    top_chunks, total_tokens_used = bm25_chunks_within_limit_sorted_low_to_high(
                        root_directory, datapoint['prefix'], datapoint['suffix'], extension
                    )
                    # Only generate descriptions for files relevant to top_chunks
                    file_descriptions = describe_files_for_top_chunks(top_chunks, description_cache)

                    # Assemble the context with file-level descriptions
                    context_parts = assemble_context(top_chunks, file_descriptions, root_directory)
                    used_tokens = 0
                    for part in context_parts:
                        print(part)
                        used_tokens += count_tokens(part)

                    total_tokens_used = used_tokens
            elif strategy == "bm25_top_5_chunks_attached_with_file_desc_refined_prompt":
                description_cache = {}
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'],
                                               extension)

                # Only generate descriptions for files relevant to top_chunks
                file_descriptions = describe_files_for_top_chunks(top_chunks, description_cache)

                # Assemble the context with file-level descriptions
                context_parts = assemble_context(top_chunks, file_descriptions, root_directory)

                used_tokens = 0
                for part in context_parts:
                    print(part)
                    used_tokens += count_tokens(part)

                total_tokens_used = used_tokens

            elif strategy == "bm25_iterative_rag_5_chunks_code_as_context":
                # Step 1: Initial retrieval
                initial_chunks = bm25_top_n_chunks(root_directory, prefix, suffix, extension)
                initial_context = ""
                for file_path, chunk in initial_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    FILE_SEP_SYMBOL_MELLUM = "<filename>"
                    initial_context += FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL,
                        file_name=clean_file_name,
                        file_content=chunk
                    )

                # Step 2: Generate rough middle code (improved prompt formatting)
                bootstrap_prompt = f"""{SYSTEM_INSTRUCTION_CODE_GEN}

                # === [EXAMPLE COMPLETION] ===
                # Example PREFIX:
                <|fim_prefix|>
                def filter_even_numbers(numbers):

                # Example SUFFIX:
                <|fim_suffix|>
                    return filtered

                Example CONTEXT:
                <|file_sep|>
                # utils/number_utils.py
                def is_even(number):
                    return number % 2 == 0
                
                <|file_sep|>
                # utils/logger_config.py
                import logging
                
                logging.basicConfig(level=logging.DEBUG)
                logger = logging.getLogger(__name__)
                <|fim_context|>


                Example COMPLETION:
                <|fim_middle|>
                filtered = []
                for number in numbers:
                    if is_even(number):
                        filtered.append(number)
                        logger.debug(f"Number  is even and added to the list.")

                # === [END EXAMPLE] ===


                # === [Relevant Code Snippets from Other Files] ===
                {initial_context}

                # === [Start of Target File] ===
                <|fim_prefix|>
                {prefix} 


                # === [CODE COMPLETION STARTS HERE] ===  
                <|fim_middle|>


                <|fim_suffix|>
                {suffix}

                # === [End of Target File] ===
                """

                try:
                    bootstrap_middle = llm_code_completion.invoke(bootstrap_prompt).strip()
                except Exception as e:
                    print(f"[Bootstrap Generation Failed]: {e}")
                    bootstrap_middle = ""

                bootstrap_middle = re.sub(r'<think>.*?</think>', '', bootstrap_middle, flags=re.DOTALL).strip()
                # Step 3: Refined retrieval
                refined_query = f"{bootstrap_middle}"
                refined_chunks = bm25_top_n_chunks(root_directory, refined_query, "", extension)
                refined_context = ""
                for file_path, chunk in refined_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    refined_context += FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL,
                        file_name=clean_file_name,
                        file_content=chunk
                    )

                # Add to context
                context_parts.append(refined_context)
                used_tokens += count_tokens(refined_context)
                total_tokens_used = used_tokens
                debug_dir = "iterative_rag_debug_dir_5_chunks_potential_code_as_2_context"
                # Optional Debug Dump
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(os.path.join(debug_dir, f"{instance_id}_initial_context.txt"), 'w',
                              encoding='utf-8') as f:
                        f.write(initial_context)
                    with open(os.path.join(debug_dir, f"{instance_id}_bootstrap_middle.txt"), 'w',
                              encoding='utf-8') as f:
                        f.write(bootstrap_middle)
                    with open(os.path.join(debug_dir, f"{instance_id}_refined_context.txt"), 'w',
                              encoding='utf-8') as f:
                        f.write(refined_context)

            elif strategy == "bm25_chunks_above_percentile":
                threshold_percentile = 98.5
                #histogram_output_dir = "bm25_histograms_practice_percentile_1.5"

                top_chunks = bm25_chunks_above_percentile(
                    root_dir=root_directory,
                    prefix=datapoint['prefix'],
                    suffix=datapoint['suffix'],
                    ext=extension,
                    percentile=threshold_percentile
                )
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL,
                        file_name=clean_file_name,
                        file_content=chunk_content
                    )
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens

            elif strategy == "recent":
                recent_filenames = datapoint['modified']
                file = find_random_recent_file(root_directory, recent_filenames)
                selected_files = [file] if file else [find_random_file(root_directory)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if strategy not in ["unix_emb_top_3_basic_chunks_512_tokens_with_metadata_reversed_order", "bm25_chunks_text_info_hint", "bm25_chunks_limited_8k", "bm25_chunks_above_percentile", "bm25_top_5_chunks_target_file_text_info", "bm25_chunks_text_info_px_sx", "bm25_chunks_text_info_mid_hint_no_other_context"
                                , "bm25_top_5_chunks_attached_with_file_desc_refined_prompt", "bm25_chunks_target_file_and_mis_code_text_info",  "bm25_top_5_chunks_methods_only_with_desc", "bm25_iterative_rag_5_chunks_code_as_context"
                    ]:
                for file_path in selected_files:
                    if not file_path:
                        continue
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        clean_file_name = file_path[len(root_directory) + 1:]
                        context_part = FILE_COMPOSE_FORMAT.format(
                            file_sep=FILE_SEP_SYMBOL,
                            file_name=clean_file_name,
                            file_content=file_content
                        )
                        context_parts.append(context_part)
                        used_tokens += count_tokens(context_part)
                    except Exception as e:
                        print(f"Skipping file {file_path} due to error: {e}")
                        continue

            context = "".join(context_parts)
            submission = {
                "context": context,
                "prefix": prefix,
                "suffix": suffix
            }

            # Add prefix/suffix if needed
            if args.trim_prefix:
                submission["prefix"] = trim_prefix(datapoint["prefix"])
            if args.trim_suffix:
                submission["suffix"] = trim_suffix(datapoint["suffix"])

            # print(f"Picked files: {[os.path.basename(f) for f in selected_files if f]}")
            # print(f"Total tokens: {used_tokens}")
            writer.write(submission)

            prefix_suffix_tokens = count_tokens(prefix) + count_tokens(suffix)

            log_token_usage(
                token_log_path=token_log_path,
                instance_id=instance_id,
                files=[os.path.basename(f) for f in selected_files if f],
                token_count=total_tokens_used,
                prefix_suffix_tokens=prefix_suffix_tokens
            )
        plot_token_usage_chart(token_log_path)