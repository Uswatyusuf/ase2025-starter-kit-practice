from dataclasses import dataclass
from typing import List
import csv
import matplotlib.pyplot as plt
import tiktoken
import random
import argparse
import os, jsonlines, concurrent.futures, re
from tqdm import tqdm
import numpy as np


from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaLLM
from chunking import basic_ast_chunk_code

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

Keep your description clear and concise (ideally 1-2 sentences).

Start your sentence with "The part requiring completion" followed by its **likely actions and objectives**.

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

Your task is to generate the missing middle code between the PREFIX and SUFFIX.
Use the CONTEXT to ensure consistency, correctness, and completeness.
"""




llm = OllamaLLM(model="qwen3:8b", temperature=0)

llm_code_completion = OllamaLLM(model="JetBrains/Mellum-4b-sft-python")

# def prepare_bm25_str(s: str) -> list[str]:
#     return "".join(c if c.isalnum() else " " for c in s.lower()).split()

# ---------- File Description Caching ----------

def load_file_description_cache(path):
    if os.path.exists(path):
        with jsonlines.open(path, 'r') as reader:
            return {r['file_path']: r['description'] for r in reader}
    return {}

def save_file_description_cache(path, description_cache):
    with jsonlines.open(path, 'w') as writer:
        for file_path, description in description_cache.items():
            writer.write({'file_path': file_path, 'description': description})

# ---------- Multi-threaded File Description Generation ----------

def batch_describe_files(file_paths, llm, max_workers=8):
    def describe(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            prompt = f"""You are a senior software developer.

Below is the full content of a file from a large codebase. 
Summarize what this file does including its main responsibilities, behaviors, and the kinds of objects/functions it defines.

Be concise but informative. 
Your response should be between (1-3) sentences.
Avoid generic phrases.
--- FILE ---
{content}
--- END FILE ---
"""
            response = llm.invoke(prompt).strip()
            # ✅ Remove anything within <think>...</think> tags
            cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            return file_path, cleaned_response
        except Exception as e:
            print(f"Failed describing {file_path}: {e}")
            return file_path, "[No description available]"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(describe, file_paths), total=len(file_paths)))
    return {file_path: desc for file_path, desc in results}


# ---------- BM25 Efficient Retrieval ----------

def bm25_top_n_chunks_with_cached_desc(
    root_dir, prefix, suffix, ext,  desc_cache_path, top_k=5, max_lines=20
):
    description_cache = load_file_description_cache(desc_cache_path)

    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r') as f:
                        if sum(1 for _ in f) > max_lines:
                            file_paths.append(file_path)
                except: pass

    uncached_files = [f for f in file_paths if f not in description_cache]
    if uncached_files:
        desc_updates = batch_describe_files(uncached_files, llm)
        description_cache.update(desc_updates)
        save_file_description_cache(desc_cache_path, description_cache)

    all_chunks = []
    for file_path in file_paths:
        desc = description_cache[file_path]
        with open(file_path, 'r') as f:
            content = f.read()
        chunks = basic_ast_chunk_code(content, language)
        all_chunks.extend([(file_path, c, desc) for c in chunks if len(c.splitlines()) > 5])

    if not all_chunks: return []

    bm25 = BM25Okapi([prepare_bm25_str(c) for _, c, _ in all_chunks])
    query = prepare_bm25_str(prefix + ' ' + suffix)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # Reverse the order to go from lowest of top 3 to highest
    reversed_top_indices = top_indices[::-1]

    return [all_chunks[i] for i in reversed_top_indices]

# ---------- Utilities ----------

def prepare_bm25_str(s):
    return re.sub(r'[^a-zA-Z0-9]', ' ', s.lower()).split()

def get_chunks_for_file(file_path: str, chunk_cache: dict, debug_dir: str = None) -> list[tuple[str, str]]:
    if file_path in chunk_cache:
        return chunk_cache[file_path]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    chunks = basic_ast_chunk_code(content, language)
    chunk_entries = [(file_path, chunk) for chunk in chunks]
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, os.path.basename(file_path) + ".chunks.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            for i, (_, chunk) in enumerate(chunk_entries):
                f.write(f"--- Chunk {i+1} ---\n{chunk.strip()}\n\n")
    chunk_cache[file_path] = chunk_entries
    return chunk_entries


def get_chunks_for_file_with_desc(
    file_path: str,
    chunk_cache: dict,
    description_cache: dict,
    debug_dir: str = None
) -> list[tuple[str, str, str]]:
    """
    Breaks a file into AST-based chunks and attaches a file-level description to each chunk.
    Returns list of (file_path, chunk_text, file_description).
    """
    if file_path in chunk_cache:
        return chunk_cache[file_path]

    # Try to read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    # Retrieve or generate file-level description
    if file_path in description_cache:
        current_file_desc = description_cache[file_path]
    else:
        current_file_desc = describe_file(file_path)
        description_cache[file_path] = current_file_desc

    current_file_desc = re.sub(r'<think>.*?</think>', '', current_file_desc, flags=re.DOTALL).strip()
    # Chunk the file and attach description
    chunks = basic_ast_chunk_code(content, language)
    chunk_entries = [(file_path, chunk, current_file_desc) for chunk in chunks]

    # Optionally write debug view of chunks
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, os.path.basename(file_path) + ".chunks.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"# 📄 File Description:\n# {current_file_desc.strip()}\n\n")
            for i, (_, chunk, _) in enumerate(chunk_entries):
                f.write(f"--- Chunk {i + 1} ---\n{chunk.strip()}\n\n")

    # Cache results
    chunk_cache[file_path] = chunk_entries
    return chunk_entries


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

    file_desc_prompt = f"""You are a senior software developer.

Below is the full content of a file from a large codebase. 
Summarize what this file does including its main responsibilities, behaviors, and the kinds of objects/functions it defines.

Be concise but informative. 
Your response should be between (1-3) sentences.
Avoid generic phrases.

--- BEGIN FILE ---
{content}
--- END FILE ---
"""
    try:
        return llm.invoke(file_desc_prompt).strip()
    except Exception as e:
        print(f"Failed to describe {file_path}: {e}")
        return "[No description available]"



def find_bm25_top_3_files(root_dir: str, prefix: str, suffix: str, min_lines: int = 10, no_of_files: int = 3) -> List[str]:
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
    reversed_top_indices = top_indices[::-1]

    return [file_names[i] for i in reversed_top_indices]


def bm25_top_n_chunks(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 5) -> list[tuple[str, str]]:
    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache))

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(prefix + " " + suffix)

    if not chunk_texts:
        return []

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    reversed_top_indices = top_indices[::-1]

    return [all_chunks[i] for i in reversed_top_indices]

def bm25_top_n_chunks_attached_with_file_desc(
    root_dir: str,
    prefix: str,
    suffix: str,
    ext: str,
    top_k: int = 3
) -> list[tuple[str, str, str]]:
    """
    Retrieves the top-k BM25-matching chunks, each with its file-level description.
    Returns a list of (file_path, chunk_content, file_description).
    """
    all_chunks = []  # type: list[tuple[str, str, str]]
    chunk_cache = {}
    description_cache = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                chunks_with_desc = get_chunks_for_file_with_desc(file_path, chunk_cache, description_cache)
                all_chunks.extend(chunks_with_desc)

    if not all_chunks:
        return []

    # Compute BM25 scores over the chunks only (ignore file descriptions for ranking)
    chunk_texts = [prepare_bm25_str(chunk) for _, chunk, _ in all_chunks]
    query = prepare_bm25_str(prefix + " " + suffix)

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)

    # Get top-k indices and reverse (if desired)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    reversed_top_indices = top_indices[::-1]

    return [all_chunks[i] for i in reversed_top_indices]


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

    enc = tiktoken.encoding_for_model("gpt-4")
    query = prepare_bm25_str(prefix + " " + suffix)
    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected_chunks = []
    context_parts = []

    # Token budget: count prefix + suffix first
    prefix_tokens = len(enc.encode(prefix))
    suffix_tokens = len(enc.encode(suffix))
    total_tokens = prefix_tokens + suffix_tokens

    for i in ranked_indices:
        file_path, chunk_content = all_chunks[i]
        clean_file_name = file_path[len(root_dir) + 1:]
        context_part = FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL,
            file_name=clean_file_name,
            file_content=chunk_content
        )

        chunk_tokens = len(enc.encode(context_part))
        if total_tokens + chunk_tokens > context_limit:
            break
        selected_chunks.append((file_path, chunk_content))
        context_parts.append(context_part)
        total_tokens += chunk_tokens
        # Sort selected chunks from low to high BM25
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

def estimate_tokens(text: str, model: str = "gpt-4")->int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

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
            elif strategy == "bm25_top_3_files":
                selected_files = find_bm25_top_3_files(root_directory, datapoint['prefix'], datapoint['suffix'])
            elif strategy == "bm25_chunks":
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += estimate_tokens(context_part)
                total_tokens_used = used_tokens
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
                    used_tokens += estimate_tokens(context_part)
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
                    used_tokens += estimate_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_target_file_text_info":
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
                    used_tokens += estimate_tokens(context_part)
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
                    used_tokens += estimate_tokens(context_part)
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
                    used_tokens += estimate_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_text_info_px_mid_sx":
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
                    middle_description = llm.invoke(fill_prompt).strip()
                    suffix_description = llm.invoke(suffix_prompt).strip()


                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    middle_description = "[LLM Description Unavailable]"
                    prefix_description = "[LLM Description Unavailable]"
                    suffix_description = "[LLM Description Unavailable]"
                prefix_description = re.sub(r'<think>.*?</think>', '', prefix_description, flags=re.DOTALL).strip()
                middle_description = re.sub(r'<think>.*?</think>', '', middle_description, flags=re.DOTALL).strip()
                suffix_description = re.sub(r'<think>.*?</think>', '', suffix_description, flags=re.DOTALL).strip()
                description_record = {
                    "instance_id": instance_id,
                    "repo": datapoint["repo"],
                    "px_description": prefix_description,
                    "mid_description": middle_description,
                    "sx_description": suffix_description
                }
                with jsonlines.open(descriptions_log_path, 'a') as descriptions_writer:
                    descriptions_writer.write(description_record)
                px_description_comment = "\n".join(f"# {line}" for line in prefix_description.splitlines())
                mid_description_comment = "\n".join(f"# {line}" for line in middle_description.splitlines())
                sx_description_comment = "\n".join(f"# {line}" for line in suffix_description.splitlines())

                # STEP 3: Prepend to prefix
                prefix = f"{px_description_comment}\n{original_prefix}"
                suffix = f"{mid_description_comment}\n\n{sx_description_comment}\n{original_suffix}"

                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += estimate_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_chunks_limited":
                    top_chunks, total_tokens_used = bm25_chunks_within_limit_sorted_low_to_high(
                        root_directory, datapoint['prefix'], datapoint['suffix'], extension
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
            elif strategy == "bm25_top_n_chunks_attached_with_file_desc":
                top_chunks = bm25_top_n_chunks_with_cached_desc(root_directory, datapoint['prefix'], datapoint['suffix'], extension, "desc_cache")
                selected_files = [file_path for file_path, _, _ in top_chunks]
                for file_path, chunk_content, file_desc in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]

                    # Attach the file description as a comment above the chunk
                    context_part = (
                        f"{FILE_SEP_SYMBOL}{clean_file_name}\n"
                        f"# This chunk comes from: {clean_file_name}\n"
                        f"# File purpose: {file_desc.strip()}\n\n"
                        f"{chunk_content.strip()}\n"
                    )
                    print(context_part)
                    context_parts.append(context_part)
                    used_tokens += estimate_tokens(context_part)
                total_tokens_used = used_tokens
            elif strategy == "bm25_iterative_rag":
                # Step 1: Initial retrieval
                initial_chunks = bm25_top_n_chunks(root_directory, prefix, suffix, extension)
                initial_context = ""
                for file_path, chunk in initial_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    FILE_SEP_SYMBOL_MELLUM = "<filename>"
                    initial_context += FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL_MELLUM,
                        file_name=clean_file_name,
                        file_content=chunk
                    )

                # Step 2: Generate rough middle code (improved prompt formatting)
                bootstrap_prompt = f"""

                # === [Relevant Code Snippets from Other Files] ===
                {initial_context}

                # === [Start of Target File] ===
                <fim_prefix> {prefix} 

                # === [COMPLETION STARTS HERE] ===
                <fim_middle>
                

                <fim_suffix> {suffix}
                # === [End of Target File] ===
                """
                try:
                    bootstrap_middle = llm_code_completion.invoke(bootstrap_prompt).strip()
                except Exception as e:
                    print(f"[Bootstrap Generation Failed]: {e}")
                    bootstrap_middle = ""

                bootstrap_middle = re.sub(r'<think>.*?</think>', '', bootstrap_middle, flags=re.DOTALL).strip()
                # Step 3: Refined retrieval
                refined_query = f"{prefix}\n{bootstrap_middle}"
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
                used_tokens += estimate_tokens(refined_context)
                total_tokens_used = used_tokens
                debug_dir = "iterative_rag_debug_dir"
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
                    used_tokens += estimate_tokens(context_part)
                total_tokens_used = used_tokens

            elif strategy == "recent":
                recent_filenames = datapoint['modified']
                file = find_random_recent_file(root_directory, recent_filenames)
                selected_files = [file] if file else [find_random_file(root_directory)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if strategy not in ["bm25_chunks_text_info_hint", "bm25_chunks_limited", "bm25_chunks_above_percentile", "bm25_chunks_target_file_text_info", "bm25_chunks_text_info_px_mid_sx", "bm25_chunks_text_info_mid_hint_no_other_context"
                                , "bm25_top_n_chunks_attached_with_file_desc", "bm25_chunks_target_file_and_mis_code_text_info"]:
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
                        used_tokens += estimate_tokens(context_part)
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

            prefix_suffix_tokens = estimate_tokens(prefix) + estimate_tokens(suffix)

            log_token_usage(
                token_log_path=token_log_path,
                instance_id=instance_id,
                files=[os.path.basename(f) for f in selected_files if f],
                token_count=total_tokens_used,
                prefix_suffix_tokens=prefix_suffix_tokens
            )
        plot_token_usage_chart(token_log_path)