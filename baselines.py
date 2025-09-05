from typing import List, Tuple
import csv
import matplotlib.pyplot as plt
import tiktoken
import random
import argparse
import os, jsonlines, re
import json
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin
import numpy as np

from tree_sitter import Language
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaLLM
from chunking import basic_ast_chunk_code, basic_ast_chunk_code_methods_only, chunk_kotlin_method_lvl_full_context

from prefix_suffix_ast_analysis import extract_prefix_nearest_block, extract_suffix_nearest_block
from prefix_suffix_ast_analysis_kt import extract_prefix_nearest_block_kt, extract_suffix_nearest_block_kt
from transformers import AutoTokenizer
from collections import defaultdict
from groq import GroqClient, AGENT_RERANK_PROMPT_TEMPLATE

PY_LANGUAGE = Language(tspython.language())
KT_LANGUAGE = Language (ts_kotlin.language())

argparser = argparse.ArgumentParser()
# Parameters for context collection strategy
argparser.add_argument("--stage", type=str, default="practice", help="Stage of the project")
argparser.add_argument("--lang", type=str, default="python", help="Language")
argparser.add_argument("--strategy", type=str, default="random", help="Context collection strategy")

# Parameters for context trimming
argparser.add_argument("--trim-prefix", action="store_true")
argparser.add_argument("--trim-suffix", action="store_true")

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



SYSTEM_INSTRUCTION_MISSING_CODE = """You are a professional developer.

Below, you are given two partial code fragments: a PREFIX and a SUFFIX from the same source file. Together, these represent parts of a larger file.

Please write a high-level description that subtly hints at what the missing middle code is likely doing.

Focus on **specific actions** the code performs, such as initializing connections, creating data structures, or invoking key functions. Include any **key purposes or behaviors**, such as validating workflows or testing performance characteristics.

Avoid using quotes, colons, or meta-language like 'Hint:'.

Start your sentence with "The part requiring completion" followed by its **likely actions and objectives**.

Keep your description clear and concise (1-2 sentences).
"""


SYSTEM_INSTRUCTION_FOR_TARGET_FILE = """You are an experienced software engineer.

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


def prepare_bm25_str(s: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in s.lower()).split()

def count_tokens(text: str) -> int:
    """
           Count the number of tokens in a given text:
               - Uses the `mellum_tokenizer` to encode the text.
               - Excludes special tokens from the count.

           :param text: Input text to tokenize.
           :return: Number of tokens in the text.
           """
    return len(mellum_tokenizer.encode(text, add_special_tokens=False))

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
        chunks = basic_ast_chunk_code(content, language)
    elif language == "kotlin":
        chunks = chunk_kotlin_method_lvl_full_context(content)
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

def find_bm25_top_n_files(root_dir: str, prefix: str, suffix: str, min_lines: int = 10, no_of_files: int = 3) -> List[str]:
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

def bm25_top_n_chunks_trimmed_px_sx(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 5, min_token_threshold: int = 2000, extra_chunk_count: int = 3) -> list[tuple[str, str]]:
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
        extracted_prefix = extract_prefix_nearest_block(px_code_bytes, caret_offset)
        extracted_suffix = extract_suffix_nearest_block(code_bytes, caret_offset)
    elif language == "kotlin":
        extracted_prefix = extract_prefix_nearest_block_kt(code_bytes, caret_offset)
        extracted_suffix = extract_suffix_nearest_block_kt(code_bytes, caret_offset)

    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache, f"debug-dir-methods-only_{language}"))

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

    # Optional: reverse if needed
    return selected_chunks[::-1]

def bm25_top_n_chunks_with_missing_code_description(root_dir: str, bm25_middle_description: str, ext: str, top_k: int = 5) -> list[tuple[str, str]]:
    """
        Retrieve top-ranked code chunks using BM25 relevance scoring:
            - Walks through the given root directory and finds files matching the extension
            - Splits files into chunks and prepares them for BM25 scoring
            - Uses the provided description as a query to rank chunks by semantic similarity
            - Returns the top N chunks in reverse-ranked order

        :param root_dir: Root directory to search for source files.
        :param bm25_middle_description: Query string used to score chunks with the BM25 algorithm.
        :param ext: File extension filter (e.g., ".py", ".kt") to include relevant files only.
        :param top_k: Number of top-ranked chunks to return (default: 5).
        :return: List of (file_path, chunk_text) tuples for the top-ranked code chunks.
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
    print(bm25_middle_description)
    query = prepare_bm25_str(bm25_middle_description)

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    reversed_top_indices = top_indices[::-1]
    return [all_chunks[i] for i in reversed_top_indices]


def bm25_chunks_within_limit_sorted_low_to_high(root_dir: str, prefix: str, suffix: str, ext: str, context_limit: int = 8000) -> tuple[list[tuple[str, str]], int]:
    """
            Retrieve BM25-ranked code chunks within a token budget,
            sorted from lowest to highest BM25 relevance.

            :param root_dir: Path to the root directory of the repository.
            :param prefix: Code snippet preceding the masked code region.
            :param suffix: Code snippet following the masked code region.
            :param ext: File extension filter for selecting files.
            :param context_limit: Maximum number of token.

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

def bm25_chunks_above_percentile(root_dir: str, prefix: str, suffix: str, ext: str, percentile: float = 90.0, histogram_dir: str = None, context_limit: int = 8000) -> list[tuple[str, str]]:
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


def assemble_context(top_chunks, file_descriptions, root_dir):
    """
        Assemble contextual snippets from top-ranked code chunks:
            - Cleans up file paths relative to the project root
            - Removes internal <think> annotations from file descriptions if any
            - Formats each chunk with file metadata and separator symbols

        :param top_chunks: List of tuples containing (file_path, chunk_content) for top-ranked chunks.
        :param file_descriptions: Dictionary mapping file paths to their descriptive summaries.
        :param root_dir: Root directory to trim from file paths for cleaner relative names.
        :return: List of formatted context parts as strings, each including file info and code content.
        """
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


def extract_first_json_object(text: str) -> str:
    """Extracts the first JSON object from a string."""
    brace_stack = []
    start_index = None

    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                start_index = i
            brace_stack.append(char)
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_index is not None:
                    return text[start_index:i+1]
    raise ValueError("No complete JSON object found in LLM response.")


def rerank_with_llm_as_judge(chunks: list[tuple[str, str]], prefix: str, suffix: str, top_k: int, groq_client) -> list[tuple[str, str]]:
    """
       Re-rank code chunks using an LLM as the ranking judge:
           - Formats candidate chunks into labeled snippets
           - Constructs a prompt with the given prefix and suffix for context
           - Sends the prompt to an LLM via the provided client for re-ranking
           - Parses the LLM’s JSON response to determine chunk order
           - Returns the top-N re-ranked chunks in reverse order

       :param chunks: List of (file_path, chunk_text) tuples representing candidate code chunks.
       :param prefix: Source code prefix used to provide left-hand context to the LLM.
       :param suffix: Source code suffix used to provide right-hand context to the LLM.
       :param top_k: Number of top re-ranked chunks to return.
       :param groq_client: Client object used to send prompts and receive responses from the LLM.
       :return: List of (file_path, chunk_text) tuples for the top-N re-ranked chunks.
       """
    formatted_snippets = []
    for i, (_, chunk) in enumerate(chunks):
        chunk_id = f"CHUNK_{i+1}"
        formatted_snippets.append(f"{chunk_id}:\n{chunk.strip()}\n")

    prompt = AGENT_RERANK_PROMPT_TEMPLATE.format(prefix, suffix, "\n".join(formatted_snippets))
    response = groq_client.formatAndSend(prompt)

    try:
        json_str = extract_first_json_object(response)
        parsed = json.loads(json_str)

        chunk_id_map = {f"CHUNK_{i + 1}": chunk for i, chunk in enumerate(chunks)}  # 👈 Move here

        print("\n [LLM RE-RANKING]:")
        for i, chunk_id in enumerate(parsed.get("files", [])):
            if chunk_id not in chunk_id_map:
                continue
            path, code = chunk_id_map[chunk_id]
            first_line = code.strip().splitlines()[0] if code.strip() else "<empty>"
            print(f"Rank {len(parsed['files']) - i}: {chunk_id} - {path}")
            print(f"Code (first line): {first_line[:80]}")

    except Exception as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {response}") from e

    ordered_chunks = []
    for chunk_id in parsed.get("files", []):
        if chunk_id not in chunk_id_map:
            print(f"[Warning] Unknown chunk ID returned by LLM: {chunk_id}")
            continue
        ordered_chunks.append(chunk_id_map[chunk_id])

    return ordered_chunks[:top_k][::-1]  # From least to top best


def bm25_llm_judge_rerank(root_dir: str, prefix: str, suffix: str, ext: str, top_k: int = 10, bm25_pool_size: int = 12) -> list[tuple[str, str]]:
    """
        Hybrid re-ranking of code chunks using BM25 and an LLM as judge:
            - Walks the given directory to collect code chunks by extension
            - Uses BM25 to retrieve the most relevant chunks based on extracted prefix/suffix context
            - Selects a candidate pool of top-ranked chunks (bm25_pool_size)
            - Ensures total tokens fit within the LLM’s context window by trimming excess chunks
            - Sends candidates to an LLM judge for final re-ranking
            - Returns the top-N re-ranked chunks

        :param root_dir: Root directory to search for source files.
        :param prefix: Source code prefix used as left-hand context around the caret position.
        :param suffix: Source code suffix used as right-hand context around the caret position.
        :param ext: File extension filter (e.g., ".py", ".kt") to include relevant files only.
        :param top_k: Number of top re-ranked chunks to return (default: 10).
        :param bm25_pool_size: Number of BM25-selected chunks to pass to the LLM for reranking (default: 12).
        :return: List of (file_path, chunk_text) tuples representing the top-N re-ranked chunks.
        """
    MAX_TOKENS_QWEN = 16000
    code_bytes = (prefix + suffix).encode("utf-8")
    px_code_bytes = prefix.encode("utf-8")
    caret_offset = len(prefix.encode("utf-8"))

    extracted_prefix = extract_prefix_nearest_block(px_code_bytes, caret_offset)
    extracted_suffix = extract_suffix_nearest_block(code_bytes, caret_offset)

    all_chunks = []
    chunk_cache = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_path = os.path.join(dirpath, filename)
                all_chunks.extend(get_chunks_for_file(file_path, chunk_cache, f"debug-dir-methods-only_{language}"))

    if not all_chunks:
        return []

    chunk_texts = [prepare_bm25_str(chunk) for _, chunk in all_chunks]
    query = prepare_bm25_str(extracted_prefix + " " + extracted_suffix)

    bm25 = BM25Okapi(chunk_texts)
    scores = bm25.get_scores(query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_candidates = top_indices[:bm25_pool_size]

    print(root_dir)
    print("\n[BM25 RANKING]:")
    for rank, idx in enumerate(bm25_candidates):
        path, code = all_chunks[idx]
        chunk_id = f"CHUNK_{rank + 1}"
        first_line = code.strip().splitlines()[0] if code.strip() else "<empty>"
        print(f"Rank {rank + 1}: {chunk_id} - {path}")
        print(f"Code (first line): {first_line[:80]}")

    # Build candidate list with top bm25_pool_size
    candidate_chunks = [all_chunks[i] for i in bm25_candidates]

    # Calculate total tokens including prefix/suffix
    total_tokens = (
        count_tokens(prefix) +
        count_tokens(suffix) +
        sum(count_tokens(chunk[1]) for chunk in candidate_chunks)
    )

    # Trim from the end until under token limit
    while total_tokens > MAX_TOKENS_QWEN and len(candidate_chunks) > 1:
        removed = candidate_chunks.pop()  # remove the lowest-ranked chunk
        removed_tokens = count_tokens(removed[1])
        total_tokens -= removed_tokens
        print(f"⚠️ Removed chunk to fit context window (now {total_tokens} tokens)")

    # Now send to Groq LLM for reranking
    groq_client = GroqClient()
    reranked = rerank_with_llm_as_judge(candidate_chunks, prefix, suffix, top_k, groq_client)
    return reranked

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
        updated_prefix = extract_prefix_nearest_block(px_code_bytes, caret_offset)
    elif language == "kotlin":
        updated_prefix = extract_prefix_nearest_block_kt(code_bytes, caret_offset)
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
        updated_suffix = extract_suffix_nearest_block(code_bytes, caret_offset)
    elif language == "kotlin":
        updated_suffix = extract_suffix_nearest_block_kt(code_bytes, caret_offset)
    return updated_suffix

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
            context_parts = []

            # Run the strategy
            if strategy == "random":
                selected_files = [find_random_file(root_directory)]
            elif strategy == "bm25":
                selected_files = [find_bm25_file(root_directory, datapoint['prefix'], datapoint['suffix'])]
            elif strategy == "bm25_top_3_files":
                selected_files = find_bm25_top_n_files(root_directory, datapoint['prefix'], datapoint['suffix'])
            elif strategy == "bm25_top_5_chunks_methods_only_with_desc":
                top_chunks= bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
            elif strategy == "bm25_top_5_2k_8_chunks_methods_only_trimmed_query_px_and_sx":
                top_chunks = bm25_top_n_chunks_trimmed_px_sx(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
            elif strategy == "llm_as_judge_top_10_chunks_methods_trimmed_query_prefix_and_suffix":
                top_chunks = bm25_llm_judge_rerank(root_directory, datapoint['prefix'], datapoint['suffix'], extension)
                selected_files = [file_path for file_path, _ in top_chunks]
                for file_path, chunk_content in top_chunks:
                    clean_file_name = file_path[len(root_directory) + 1:]
                    context_part = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                              file_content=chunk_content)
                    context_parts.append(context_part)
            elif strategy == "bm25_chunks_middle_code_desc_as_bm25_query":
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
            elif strategy == "bm25_top_5_chunks_target_file_text_info":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_FOR_TARGET_FILE}

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
            elif strategy == "bm25_chunks_text_info_px_sx":
                # STEP 1: Get original prefix/suffix
                original_prefix = datapoint["prefix"]
                original_suffix = datapoint["suffix"]

                # STEP 2: Query LLM for a description
                fill_prompt = f"""{SYSTEM_INSTRUCTION_MISSING_CODE}

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
                    suffix_description = llm.invoke(suffix_prompt).strip()


                except Exception as e:
                    print(f"LLM failed for instance {instance_id}: {e}")
                    prefix_description = "[LLM Description Unavailable]"
                    suffix_description = "[LLM Description Unavailable]"
                prefix_description = re.sub(r'<think>.*?</think>', '', prefix_description, flags=re.DOTALL).strip()
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
            elif strategy == "bm25_top_5_chunks_attached_with_file_desc_refined_prompt":
                description_cache = {}
                top_chunks = bm25_top_n_chunks(root_directory, datapoint['prefix'], datapoint['suffix'],
                                               extension)

                # Only generate descriptions for files relevant to top_chunks
                file_descriptions = describe_files_for_top_chunks(top_chunks, description_cache)

                # Assemble the context with file-level descriptions
                context_parts = assemble_context(top_chunks, file_descriptions, root_directory)
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

                # Step 2: Generate rough middle code
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
            elif strategy == "bm25_chunks_above_percentile":
                threshold_percentile = 98.5

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
            elif strategy == "recent":
                recent_filenames = datapoint['modified']
                file = find_random_recent_file(root_directory, recent_filenames)
                selected_files = [file] if file else [find_random_file(root_directory)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # add all strategies involving chunks to the list
            if strategy not in [ "bm25_top_5_2k_8_chunks_methods_only_trimmed_query_px_and_sx"]:
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
            writer.write(submission)


