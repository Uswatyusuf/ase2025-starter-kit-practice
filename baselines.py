from typing import List
import csv
import matplotlib.pyplot as plt
import os
import tiktoken
import json
import re
import random
import argparse
import jsonlines
import numpy as np
import langchain_ollama

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

SYSTEM_INSTRUCTION_MID = """You are a professional software developer.

You are given a PREFIX and a SUFFIX from a source file. Your job is to generate a short, subtle hint that nudges a fellow developer about what the missing code likely does.

Do not describe it fully. Do not repeat "the missing code". Just write a hint in natural, suggestive language — like a short inline comment.

Examples:
- Likely uses a loop to publish messages with priorities.
- May publish messages using kombu.Producer with retry enabled.
- Probably drains events and verifies the order.

Keep it neutral, 2 short sentence, starting with 'Hint:'.
"""

SYSTEM_INSTRUCTION_W_TARGET_FILE = """You are an experienced software engineer.

Below, you are given two partial code fragments: a PREFIX and a SUFFIX from the same source file. Together, these represent parts of a larger file.

Please write a brief, high-level description of the **purpose** of this file, based on the given code. Focus on describing what the file is supposed to do overall (its main functionality or role in the project).

Keep your description short and clear (ideally 1-3 sentences).

"""


llm = OllamaLLM(model="qwen3:1.7b")


def prepare_bm25_str(s: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in s.lower()).split()


def get_chunks_for_file(file_path: str, chunk_cache: dict, debug_dir: str = None) -> list[tuple[str, str]]:
    if file_path in chunk_cache:
        return chunk_cache[file_path]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    chunks = basic_ast_chunk_code(content)
    chunk_entries = [(file_path, chunk) for chunk in chunks]
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, os.path.basename(file_path) + ".chunks.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            for i, (_, chunk) in enumerate(chunk_entries):
                f.write(f"--- Chunk {i+1} ---\n{chunk.strip()}\n\n")
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
            elif strategy == "bm25_chunks_target_file_text_info":
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
                prefix = f"{original_prefix}\n\n{description_comment}"
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

            if strategy not in ["bm25_chunks_text_info_hint", "bm25_chunks_limited", "bm25_chunks_above_percentile"]:
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