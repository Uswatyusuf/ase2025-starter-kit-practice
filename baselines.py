from typing import List
import os

import random
import argparse
import jsonlines
import numpy as np

from rank_bm25 import BM25Okapi
from chunking import chunk_python, chunk_kotlin_method_lvl
from prefix_suffix_ast_analysis_kt import extract_prefix_from_last_block_kt, extract_suffix_to_next_block_kt
from prefix_suffix_ast_analysis_py import extract_prefix_from_last_block, extract_suffix_to_next_block
from transformers import AutoTokenizer

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

mellum_tokenizer = AutoTokenizer.from_pretrained("JetBrains/Mellum-4b-sft-python", trust_remote_code=True, use_fast=False)

def prepare_bm25_str(s: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in s.lower()).split()

def get_chunks_for_file(file_path: str, chunk_cache: dict, debug_dir: str = None) -> list[tuple[str, str]]:
    """
        Retrieve or generate code chunks for a given source file.

        This function reads the specified source file and splits it into code chunks
        using language-specific chunking functions. It caches the results in `chunk_cache` to avoid
        redundant computation.

        :param file_path: Path to the source code file.
        :param chunk_cache: A cache mapping file paths to their list of chunks.
        :param debug_dir: If provided, writes the generated chunks to
            a `.chunks.txt` file in this directory for debugging purposes.

        :return: List of tuples representing the file path and each code chunk extracted from the file.
        """
    if file_path in chunk_cache:
        return chunk_cache[file_path]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    if language == "python":
        chunks = chunk_python(content)
    elif language == "kotlin":
        chunks = chunk_kotlin_method_lvl(content)
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


def find_bm25_top_n_files(root_dir: str, prefix: str, suffix: str, min_lines: int = 10, no_of_files: int = 4) -> List[str]:
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
    :return: List of tuples representing the file path
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
                    pass

    query = (prefix + " " + suffix).lower()
    query = prepare_bm25_str(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:no_of_files]

    # Reverse the order to go from lowest of top 3 to highest
    reversed_top_indices = top_indices[::-1]

    return [file_names[i] for i in reversed_top_indices]


def bm25_top_n_chunks(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 3) -> list[tuple[str, str]]:
    """
        Retrieve the top-N most relevant code chunks from a repository using BM25 ranking.
         - in the given language
        - with the highest BM25 score with the completion file
        - in the given directory and its subdirectories


        :param root_dir: Path to the root directory of the repository to search.
        :param prefix: Code snippet immediately preceding the masked region.
        :param suffix: Code snippet immediately following the masked region.
        :param ext: File extension filter.
        :param top_k : top-ranked chunks to return, defaults to 3.

        :return: List of tuples representing the file path and chunk content of the top-ranked code chunks
        """
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

def bm25_top_n_chunks_trimmed_px_sx(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 8) -> list[tuple[str, str]]:
    """
           Retrieve the top-N most relevant code chunks from a repository using BM25 ranking.
            - in the given language
           - with the highest BM25 score with the completion file
           - in the given directory and its subdirectories
           - with trimmed prefix and suffix


           :param root_dir: Path to the root directory of the repository to search.
           :param prefix: Code snippet immediately preceding the masked region.
           :param suffix: Code snippet immediately following the masked region.
           :param ext: File extension filter.
           :param top_k : top-ranked chunks to return, defaults to 3.

           :return: List of tuples representing the file path and chunk content of the top-ranked code chunks

   """
    global extracted_prefix, extracted_suffix

    code_bytes = (prefix + suffix).encode("utf-8")
    px_code_bytes = prefix.encode("utf-8")
    caret_offset = len(prefix.encode("utf-8"))

    if language == "python":
        extracted_prefix = extract_prefix_from_last_block(px_code_bytes, caret_offset)
        extracted_suffix = extract_suffix_to_next_block(code_bytes, caret_offset)
    elif language == "kotlin":
        extracted_prefix = extract_prefix_from_last_block_kt(code_bytes, caret_offset)
        extracted_suffix = extract_suffix_to_next_block_kt(code_bytes, caret_offset)

    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache))

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(extracted_prefix + " " + extracted_suffix)

    if not chunk_texts:
        return []

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected_indices = ranked_indices[:top_k]
    selected_chunks = [all_chunks[i] for i in selected_indices]
    return selected_chunks[::-1]

def bm25_top_n_chunks_2k_min_trimmed_px_sx(
    root_dir: str,
    prefix: str,
    suffix: str,
    ext: str,
    top_k: int = 5,
    min_token_threshold: int = 2000,
    extra_chunk_count: int = 3
) -> list[tuple[str, str]]:
    """
       Retrieve top-ranked code chunks using BM25 ranking with prefix/suffix trimming
       and a minimum token threshold.
         3. Ensuring that the combined token count of the selected chunks meets a
            minimum threshold (`min_token_threshold`). If the threshold is not
            reached, additional chunks are added incrementally.


       :param  root_dir: Root directory of the repository to search for code files.
       :param  prefix: Code snippet immediately preceding the masked code region.
       :param  suffix: Code snippet immediately following the masked code region.
       :param  ext: File extension.
       :param  top_k: Initial number of top-ranked chunks to select. Defaults to 5.
       :param    min_token_threshold : Minimum total token count required
               across selected chunks. Defaults to 2000.
       :param    extra_chunk_count: Number of additional chunks to add
               if token threshold is not met. Defaults to 3.

       :return: List of tuples representing the file path and chunk content of the top-ranked code chunks
       """
    global extracted_prefix, extracted_suffix

    code_bytes = (prefix + suffix).encode("utf-8")
    px_code_bytes = prefix.encode("utf-8")
    caret_offset = len(prefix.encode("utf-8"))

    if language == "python":
        extracted_prefix = extract_prefix_from_last_block(px_code_bytes, caret_offset)
        extracted_suffix = extract_suffix_to_next_block(code_bytes, caret_offset)
    elif language == "kotlin":
        extracted_prefix = extract_prefix_from_last_block_kt(code_bytes, caret_offset)
        extracted_suffix = extract_suffix_to_next_block_kt(code_bytes, caret_offset)

    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache))
    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(extracted_prefix + " " + extracted_suffix)
    if not chunk_texts:
        return []

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Start with top_k chunks
    selected_indices = ranked_indices[:top_k]
    selected_chunks = [all_chunks[i] for i in selected_indices]

    # Count total tokens
    total_tokens = sum(count_tokens(chunk[1]) for chunk in selected_chunks)

    # Add more chunks if below threshold
    if total_tokens < min_token_threshold:
        additional_indices = ranked_indices[top_k : top_k + extra_chunk_count]
        selected_indices.extend(additional_indices)
        selected_chunks = [all_chunks[i] for i in selected_indices]
    return selected_chunks[::-1]

def bm25_chunks_within_limit_sorted_low_to_high(
    root_dir: str,
    prefix: str,
    suffix: str,
    ext: str,
    context_limit: int = 8000
) -> tuple[list[tuple[str, str]], int]:
    """
        Retrieve BM25-ranked code chunks within a token budget,
        sorted from lowest to highest BM25 relevance.

        :param root_dir: Path to the root directory of the repository.
        :param prefix: Code snippet preceding the masked code region.
        :param suffix: Code snippet following the masked code region.
        :param ext: File extension filter for selecting files.
        :param context_limit: Maximum number of token.
           000.

        :return: tuple[list[tuple[str, str]], int]:
                - List of (file_path, chunk) tuples representing the selected chunks.
                - total number of tokens used (including prefix, suffix, and chunks).
        """
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

    # Token budget: count prefix + suffix first
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
    context_limit: int = 8000,
) -> list[tuple[str, str]]:
    """
       Select code chunks above a BM25 score percentile threshold:
           - Searches all files with the given extension in the repository.
           - Splits files into code chunks and ranks them using BM25.
           - Selects only chunks with scores at or above the specified percentile.
           - Greedily adds top-ranked chunks until the context token limit is reached.

       :param root_dir: Root directory of the repository to search.
       :param prefix: Code snippet preceding the masked code region.
       :param suffix: Code snippet following the masked code region.
       :param ext: File extension filter (e.g., ".py", ".kt") for selecting files.
       :param percentile: Percentile threshold for selecting top BM25-scored chunks.
                          Defaults to 90.0.
       :param context_limit: Maximum total token budget for selected chunks (including
                             prefix and suffix). Defaults to 8000.
       :return: A list of (file_path, chunk_content) tuples for chunks above the percentile
                threshold, sorted by BM25 relevance score (highest first).
       """
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


    total_tokens = count_tokens(prefix) + count_tokens(suffix)

    # Sort by BM25 score high → low for greedy selection
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)

    included_chunks = []
    for (file_path, chunk_content), _ in selected_sorted:
        context_part = FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL,
            file_name=file_path[len(root_dir) + 1:],
            file_content=chunk_content
        )
        chunk_tokens = count_tokens(context_part)
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
    """
       Extract a trimmed and focused code prefix based on language-specific block boundaries:
           - Determines the caret position based on the length of the prefix.
           - Uses language-specific functions to extract the closest enclosing block

       :param prefix: Code snippet preceding the masked code region.
       :return: Trimmed prefix string extracted from the nearest logical code block.
       """
    global updated_prefix
    code_bytes = (prefix + suffix).encode("utf-8")
    px_code_bytes = prefix.encode("utf-8")
    caret_offset = len(prefix.encode("utf-8"))
    if language == "python":
        updated_prefix = extract_prefix_from_last_block(px_code_bytes, caret_offset)
    elif language == "kotlin":
        updated_prefix = extract_prefix_from_last_block_kt(code_bytes, caret_offset)
    return updated_prefix

def trim_suffix(suffix: str):
    """
        Extract a trimmed and focused code suffix based on language-specific block boundaries:
            - Determines the caret position based on the prefix length.
            - Uses language-specific functions to extract the next enclosing block

        :param suffix: Code snippet following the masked code region.
        :return: Trimmed suffix string extracted from the nearest logical code block.
        """
    global  updated_suffix
    code_bytes = (prefix + suffix).encode("utf-8")
    caret_offset = len(prefix.encode("utf-8"))
    if language == "python":
        updated_suffix = extract_suffix_to_next_block(code_bytes, caret_offset)
    elif language == "kotlin":
        updated_suffix = extract_suffix_to_next_block_kt(code_bytes, caret_offset)
    return updated_suffix

def count_tokens(text: str) -> int:
    """
        Count the number of tokens in a given text:
            - Uses the `mellum_tokenizer` to encode the text.
            - Excludes special tokens from the count.

        :param text: Input text to tokenize.
        :return: Number of tokens in the text.
        """
    return len(mellum_tokenizer.encode(text, add_special_tokens=False))

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
                selected_files = find_bm25_top_n_files(root_directory, datapoint['prefix'], datapoint['suffix'])
            elif strategy == "bm25_chunks":
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
                    used_tokens += count_tokens(context_part)
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
                    used_tokens += count_tokens(context_part)
                total_tokens_used = used_tokens

            elif strategy == "recent":
                recent_filenames = datapoint['modified']
                file = find_random_recent_file(root_directory, recent_filenames)
                selected_files = [file] if file else [find_random_file(root_directory)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if strategy not in ["bm25_chunks", "bm25_chunks_limited", "bm25_chunks_above_percentile"]:
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
            submission = {"context": context}

            # Add prefix/suffix if needed
            if args.trim_prefix:
                submission["prefix"] = trim_prefix(datapoint["prefix"])
            if args.trim_suffix:
                submission["suffix"] = trim_suffix(datapoint["suffix"])

            writer.write(submission)


