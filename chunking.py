from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin
from transformers import AutoTokenizer
from langchain_ollama import OllamaLLM
import re

file_path = "data/repositories-python-practice/celery__kombu-7f9674419b585921b1da4ecbd5f3dc203891955e/kombu/utils/eventio.py"
mellum_tokenizer = AutoTokenizer.from_pretrained("JetBrains/Mellum-4b-sft-python")
#tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
MAX_TOKENS = 512

llm = OllamaLLM(model="qwen3:8b", temperature=0)

SYSTEM_INSTRUCTION_METHOD_DESC = """You are an experienced software engineer.

Below, you are given two partial code fragments: a PREFIX and a SUFFIX from the same source file. Together, these represent parts of a larger file.

Please write a brief, high-level description of the **purpose** of this file, based on the given code. Focus on describing what the file is supposed to do overall (its main functionality or role in the project).

Keep your description short and clear (ideally 1-3 sentences).

"""

# Read the actual file content
with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

PY_LANGUAGE = Language(tspython.language())
KT_LANGUAGE = Language (ts_kotlin.language())

# def cast_chunk_code(code: str, max_chunk_size: int = 1000) -> list[str]:
#     PY_LANGUAGE = Language(tspython.language())
#     parser = Parser(PY_LANGUAGE)
#
#
#     tree = parser.parse(code.encode("utf-8"))
#     root = tree.root_node
#     code_bytes = code.encode("utf-8")
#
#     def get_node_size(node):
#         return len(code_bytes[node.start_byte:node.end_byte].replace(b" ", b"").replace(b"\n", b""))
#
#     def chunk_ast_nodes(nodes):
#         chunks = []
#         chunk = []
#         size = 0
#
#         for node in nodes:
#             node_size = get_node_size(node)
#             if node_size > max_chunk_size:
#                 if chunk:
#                     chunks.append(chunk)
#                     chunk = []
#                     size = 0
#                 subchunks = chunk_ast_nodes(node.children)
#                 chunks.extend(subchunks)
#             elif size + node_size > max_chunk_size:
#                 chunks.append(chunk)
#                 chunk = [node]
#                 size = node_size
#             else:
#                 chunk.append(node)
#                 size += node_size
#
#         if chunk:
#             chunks.append(chunk)
#         return chunks
#
#     ast_chunks = chunk_ast_nodes(root.children)
#     return [
#         "".join([code_bytes[n.start_byte:n.end_byte].decode("utf-8") for n in chunk])
#         for chunk in ast_chunks
#     ]

def summarize_class_purpose(class_name, full_code):
    prompt = f"""
    Given the following code, summarize in one sentence what the class '{class_name}' does. Be concise and avoid restating the class name unnecessarily.

    Code:
    {full_code}
    """
    response = llm(prompt)
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return clean_response

def chunk_class_methods_with_desc(node, code_bytes, max_tokens, class_purpose):
    """Chunk class by methods, preserving header, docstring, and adding LLM-generated purpose."""
    chunks = []
    header = extract_class_header(node, code_bytes)
    methods = [c for c in node.children if c.type == "function_definition"]

    class_name = "Unknown"
    for child in node.children:
        if child.type == "identifier":
            class_name = extract_node_text(code_bytes, child)
            break

    current_chunk = header
    current_token_count = token_count(current_chunk)
    included_methods = []

    def flush_chunk():
        if included_methods:
            methods_comment = f"# Methods included: {', '.join(included_methods)}\n"
            purpose_comment = f"# Class: {class_name} (partial)\n# Purpose: {class_purpose}\n" + methods_comment
            final_chunk = purpose_comment + current_chunk.rstrip()
            chunks.append(final_chunk)

    for method in methods:
        method_text = extract_node_text(code_bytes, method)
        method_text_clean = clean_chunk_text(method_text)
        method_tokens = token_count(method_text_clean)
        method_name = "unknown_method"
        for child in method.children:
            if child.type == "identifier":
                method_name = extract_node_text(code_bytes, child)
                break

        if current_token_count + method_tokens > max_tokens:
            flush_chunk()
            current_chunk = header + method_text_clean + "\n"
            current_token_count = token_count(current_chunk)
            included_methods = [method_name]
        else:
            current_chunk += method_text_clean + "\n"
            current_token_count += method_tokens
            included_methods.append(method_name)

    if included_methods:
        flush_chunk()
    return chunks


def basic_ast_chunk_code(code: str, lang: str) -> list[str]:
    max_tokens = 500
    global parser
    if lang == "python":
        parser = Parser(PY_LANGUAGE)

    elif lang == "kotlin":
        parser = Parser(PY_LANGUAGE)


    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    chunks = []
    current_expr_group = []

    def flush_expr_group():
        if current_expr_group:
            chunk_text = "\n".join(
                code_bytes[n.start_byte:n.end_byte].decode("utf-8").strip()
                for n in current_expr_group
            )
            if chunk_text.strip():
                chunks.append(chunk_text)
            current_expr_group.clear()

    for node in root.children:
        if node.type in {"future_import_statement", "import_statement", "import_from_statement", "import_header"}:
            continue

        if node.type == "expression_statement":
            current_expr_group.append(node)
        elif node.type == "class_definition":
            node_text = extract_node_text(code_bytes, node)
            node_tokens = token_count(node_text)
            class_name = "Unknown"
            for child in node.children:
                if child.type == "identifier":
                    class_name = extract_node_text(code_bytes, child)
                    break

            class_purpose = summarize_class_purpose(class_name, code)
            print(class_purpose)

            if node_tokens > max_tokens:
                class_chunks = chunk_class_methods_with_desc(node, code_bytes, max_tokens, class_purpose)
                chunks.extend(class_chunks)
            else:
                purpose_comment = f"# Class: {class_name}\n# Purpose: {class_purpose}\n"
                chunks.append(purpose_comment + node_text)
        else:
            flush_expr_group()
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            if chunk_text.strip():
                chunks.append(chunk_text)

    flush_expr_group()
    return chunks


def basic_ast_chunk_code_methods_only(code: str, lang: str) -> list[str]:
    global parser
    print("Language; ", lang)
    if lang == "python":
        parser = Parser(PY_LANGUAGE)
    elif lang == "kotlin":
        parser = Parser(KT_LANGUAGE)
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    chunks = []

    def extract_with_decorators(code_bytes, node):
        """Extract node text including decorators directly attached to it."""
        decorators = []
        idx = node.prev_sibling
        while idx and idx.type == "decorator":
            decorators.insert(0, extract_node_text(code_bytes, idx).rstrip())
            idx = idx.prev_sibling
        node_text = extract_node_text(code_bytes, node).rstrip()
        return "\n".join(decorators + [node_text])

    def extract_class_decorators(code_bytes, class_node):
        """Extract decorators attached to the class itself."""
        decorators = []
        idx = class_node.prev_sibling
        while idx and idx.type == "decorated_definition":
            decorators.insert(0, extract_node_text(code_bytes, idx).rstrip())
            idx = idx.prev_sibling
        return "\n".join(decorators).rstrip()

    def extract_class_header_methods_only(node, code_bytes):
        """Extract only the class header (without methods)."""
        # From start of class_node to the colon (:), typically first line
        # Use node.start_byte to node.body.start_byte if available
        class_name = ""
        for child in node.children:
            if child.type == "identifier":
                class_name = extract_node_text(code_bytes, child)
                break
        return f"class {class_name}:"

    for node in root.children:
        if node.type in {"future_import_statement", "import_statement", "import_from_statement", "import_header"}:
            continue
        if node.type == "decorated_definition":
            print("Decorated function here!!")
            for c in node.children:
                if c.type == "class_definition":
                    class_decorators = extract_class_decorators(code_bytes, node)
                    class_header = extract_class_header_methods_only(c, code_bytes)
                    for c_ch in c.children:
                        if c_ch.type == "block":
                            for c_ch_ch in c_ch.children:
                                methods = [c_ch_ch_ch for c_ch_ch_ch in c_ch.children if
                                           c_ch_ch.type == "function_definition"]
                                print("Decorated function here!!-----chunked")
                                for method in methods:
                                    method_text = extract_with_decorators(code_bytes, method)
                                    combined = "\n".join(filter(None, [class_decorators, class_header, method_text]))
                                    chunks.append(combined)

        # Handle class definitions by methods
        if node.type == "class_definition":
            class_decorators = extract_class_decorators(code_bytes, node)
            class_header = extract_class_header_methods_only(node, code_bytes)

            methods = [c for c in node.children if c.type == "function_definition"]

            for method in methods:
                method_text = extract_with_decorators(code_bytes, method)
                combined = "\n".join(filter(None, [class_decorators, class_header, method_text]))
                chunks.append(combined)

        elif node.type == "function_definition":
            # Handle global functions
            func_text = extract_with_decorators(code_bytes, node)
            chunks.append(func_text)
        else:
            # Other non-import non-class nodes
            node_text = extract_node_text(code_bytes, node)
            if node_text.strip():
                chunks.append(node_text)

    return chunks

def print_ast_structure(code: str):
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    print("Root type:", root.type)

    for i, node in enumerate(root.children):
        node_type = node.type
        node_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8").strip()
        print(f"--- Node {i+1} ---")
        print(f"Type: {node_type}")
        print(f"Text:\n{node_text}")
        print()


def token_count(text: str) -> int:
    return len(mellum_tokenizer.encode(text, add_special_tokens=False))

def clean_chunk_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line.strip())
    return cleaned.strip()

def extract_node_text(code_bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

def extract_class_header(node, code_bytes) -> str:
    name = None
    docstring = None
    for child in node.children:
        if child.type == "identifier":
            name = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "string" and docstring is None:
            docstring = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
    header = f"class {name}:\n" if name else "class Unknown:\n"
    if docstring:
        # indent docstring
        docstring_lines = docstring.splitlines()
        docstring_indented = "\n".join("    "+line for line in docstring_lines)
        header += docstring_indented + "\n"
    header += "    # Continued from previous chunk\n"
    return header

def chunk_function_body(node, code_bytes, max_tokens):
    """Chunk function body by statements if too large."""
    chunks = []
    header = ""
    name = None
    for child in node.children:
        if child.type == "identifier":
            name = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
            break
    header = f"def {name}(...):\n" if name else "def unknown_function(...):\n"

    stmts = [c for c in node.children if c.type != "identifier" and c.type != "parameters"]
    current_chunk = header
    current_token_count = token_count(current_chunk)

    for stmt in stmts:
        stmt_text = extract_node_text(code_bytes, stmt)
        stmt_text_clean = clean_chunk_text(stmt_text)
        stmt_tokens = token_count(stmt_text_clean)
        if current_token_count + stmt_tokens > max_tokens:
            chunks.append(current_chunk.rstrip())
            current_chunk = header + stmt_text_clean + "\n"
            current_token_count = token_count(current_chunk)
        else:
            current_chunk += stmt_text_clean + "\n"
            current_token_count += stmt_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.rstrip())
    return chunks

def chunk_class_methods(node, code_bytes, max_tokens):
    """Chunk class by methods, preserving header and docstring."""
    chunks = []
    header = extract_class_header(node, code_bytes)
    methods = [c for c in node.children if c.type == "function_definition"]

    current_chunk = header

    for method in methods:
        method_text = extract_node_text(code_bytes, method)
        method_text_clean = clean_chunk_text(method_text)
        current_chunk += method_text_clean + "\n"


    if current_chunk.strip():
        chunks.append(current_chunk.rstrip())
    return chunks

def ast_chunk_code_with_comments(code: str, lang: str, max_tokens=MAX_TOKENS) -> list[str]:
    global parser
    if lang == "python":
        parser = Parser(PY_LANGUAGE)
    elif lang == "kotlin":
        parser = Parser(KT_LANGUAGE)
    else:
        raise ValueError(f"Unsupported language: {lang}")

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    chunks = []
    current_expr_group = []
    current_token_count = 0

    def flush_expr_group():
        nonlocal current_token_count
        if not current_expr_group:
            return

        chunk_text = "\n".join(
            clean_chunk_text(extract_node_text(code_bytes, n))
            for n in current_expr_group
        )
        tokens = token_count(chunk_text)
        if current_token_count + tokens > max_tokens:
            # flush current chunk before adding new
            if chunks and current_token_count > 0:
                chunks.append(chunks.pop())
            chunks.append(chunk_text)
            current_token_count = tokens
        else:
            if chunks:
                chunks[-1] += "\n" + chunk_text
            else:
                chunks.append(chunk_text)
            current_token_count += tokens
        current_expr_group.clear()

    for node in root.children:
        # Skip imports
        if node.type in {"future_import_statement", "import_statement", "import_from_statement"}:
            continue

        if node.type == "expression_statement":
            current_expr_group.append(node)
            # Flush if expr group too big
            if current_token_count + token_count(extract_node_text(code_bytes, node)) > max_tokens:
                flush_expr_group()

        else:
            flush_expr_group()

            node_text = extract_node_text(code_bytes, node)
            node_text_clean = clean_chunk_text(node_text)
            node_tokens = token_count(node_text_clean)

            if node.type == "function_definition":
                if node_tokens > max_tokens:
                    # Chunk function body internally
                    func_chunks = chunk_function_body(node, code_bytes, max_tokens)
                    chunks.extend(func_chunks)
                else:
                    name = None
                    for child in node.children:
                        if child.type == "identifier":
                            name = extract_node_text(code_bytes, child)
                            break
                    header = f"# Function: {name}\n" if name else "# Function\n"
                    chunks.append(header + node_text_clean)
            elif node.type == "class_definition":
                if node_tokens > max_tokens:
                    class_chunks = chunk_class_methods(node, code_bytes, max_tokens)
                    chunks.extend(class_chunks)
                else:
                    name = None
                    for child in node.children:
                        if child.type == "identifier":
                            name = extract_node_text(code_bytes, child)
                            break
                    header = f"# Class: {name}\n" if name else "# Class\n"
                    chunks.append(header + node_text_clean)
            else:
                # Other top level nodes
                chunks.append(f"# {node.type.replace('_', ' ').capitalize()}\n{node_text_clean}")

    flush_expr_group()
    return chunks
