from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin
from transformers import AutoTokenizer
from langchain_ollama import OllamaLLM
import re
import ollama



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

def summarize_class_with_qwen(class_code: str) -> str:
    """Ask Qwen to summarize the purpose of a class with no reasoning."""
    prompt = f"""
You are an assistant that summarizes Python classes literally.
Do not infer any purpose not explicitly shown in code.

Class:
\"\"\"
{class_code}
\"\"\"

Give a one-sentence summary of what this class does. Do not explain or reason.
"""
    response = ollama.chat(
        model='qwen3:8b',
        messages=[{"role": "user", "content": prompt}]
    )
    return re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL).strip()


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


def basic_ast_chunk_code(code: str) -> list[str]:
    PY_LANGUAGE = Language(tspython.language())
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
        # Skip import statements
        if node.type in {"future_import_statement", "import_statement", "import_from_statement"}:
            continue

        if node.type == "expression_statement":
            current_expr_group.append(node)
        else:
            flush_expr_group()
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            if chunk_text.strip():
                chunks.append(chunk_text)

    flush_expr_group()
    return chunks



def basic_ast_chunk_code_methods_only(code: str, lang: str) -> list[str]:
    global parser
    if lang.lower() == "python":
        parser = Parser(PY_LANGUAGE)
    elif lang.lower() == "kotlin":
        parser = Parser(KT_LANGUAGE)

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node
    chunks = []

    def extract_node_text(node):
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def extract_decorators_above(node):
        decorators = []
        sibling = node.prev_sibling
        while sibling and sibling.type == "decorator":
            decorators.insert(0, extract_node_text(sibling).rstrip())
            sibling = sibling.prev_sibling
        return decorators

    def extract_class_header(class_node):
        class_name_node = class_node.child_by_field_name("name")
        class_name = extract_node_text(class_name_node)
        return f"class {class_name}:"

    def process_class(class_node, class_decorators):
        class_header = extract_class_header(class_node)
        body = class_node.child_by_field_name("body")
        if not body:
            return
        for item in body.children:
            if item.type == "function_definition":
                method_decorators = extract_decorators_above(item)
                method_text = extract_node_text(item).rstrip()
                full_chunk = "\n".join(
                    local_imports + class_decorators + [class_header] + method_decorators + [method_text]
                )
                chunks.append(full_chunk)

    # 1. Collect top-level imports
    local_imports = []
    for node in root.children:
        if node.type in {"import_statement", "import_from_statement", "future_import_statement"}:
            local_imports.append(extract_node_text(node).strip())

    # 2. Re-process and extract method/function/class chunks
    for node in root.children:
        if node.type in {"import_statement", "import_from_statement", "future_import_statement"}:
            continue

        if node.type == "decorated_definition":
            decorators = extract_decorators_above(node)
            inner = next(
                (child for child in node.children if child.type in {"class_definition", "function_definition"}), None
            )
            if inner and inner.type == "class_definition":
                process_class(inner, decorators)
            elif inner and inner.type == "function_definition":
                method_decorators = extract_decorators_above(inner)
                full_text = "\n".join( decorators + method_decorators + [extract_node_text(inner).rstrip()])
                chunks.append(full_text)

        elif node.type == "class_definition":
            process_class(node, [])

        elif node.type == "function_definition":
            method_decorators = extract_decorators_above(node)
            full_text = "\n".join(method_decorators + [extract_node_text(node).rstrip()])
            chunks.append(full_text)

        else:
            node_text = extract_node_text(node).strip()
            if node_text:
                full_chunk = "\n".join([node_text])
                chunks.append(full_chunk)

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


def chunk_kotlin_with_full_context(code: str) -> list[str]:
    parser = Parser(KT_LANGUAGE)
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node
    chunks = []

    def extract_text(node):
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def get_annotations_and_modifiers(node):
        items = []
        sibling = node.prev_sibling
        while sibling:
            if sibling.type in {"annotation", "modifiers"}:
                items.insert(0, extract_text(sibling).strip())
            elif sibling.type == "comment" and extract_text(sibling).strip().startswith("/**"):
                items.insert(0, extract_text(sibling).strip())
            else:
                break
            sibling = sibling.prev_sibling
        return items

    def get_parent_context(node):
        context = []
        parent = node.parent
        while parent and parent != root:
            if parent.type in {"class_declaration", "object_declaration", "interface_declaration"}:
                context = (
                    get_annotations_and_modifiers(parent)
                    + [extract_text(parent).split("{")[0].strip()]
                    + context
                )
            elif parent.type == "companion_object":
                context = ["companion object"] + context
            parent = parent.parent
        return context

    def get_package_and_imports():
        headers = []
        for node in root.children:
            if node.type in {"package_header", "import_list", "import_header", "import_statement"}:
                headers.append(extract_text(node).strip())
        return headers

    def process_function(node):
        if node.type != "function_declaration":
            return
        annotations = get_annotations_and_modifiers(node)
        context = get_parent_context(node)
        method_body = extract_text(node).strip()
        full = context + annotations + [method_body]
        chunks.append("\n".join(full))

    def walk(node):
        if node.type == "function_declaration":
            process_function(node)
        for child in node.children:
            walk(child)

    walk(root)
    return chunks

def basic_ast_chunk_code_methods_only_with_desc(code: str, lang: str) -> list[str]:
    if lang.lower() == "python":
        parser = Parser(PY_LANGUAGE)
    elif lang.lower() == "kotlin":
        parser = Parser(KT_LANGUAGE)
    else:
        raise ValueError("Unsupported language")

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node
    chunks = []

    def extract_node_text(node):
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def extract_decorators_above(node):
        decorators = []
        sibling = node.prev_sibling
        while sibling and sibling.type == "decorator":
            decorators.insert(0, extract_node_text(sibling).rstrip())
            sibling = sibling.prev_sibling
        return decorators

    def extract_class_name(class_node):
        name_node = class_node.child_by_field_name("name")
        return extract_node_text(name_node) if name_node else "UnknownClass"

    # 1. Extract top-level imports
    local_imports = []
    for node in root.children:
        if node.type in {"import_statement", "import_from_statement", "future_import_statement"}:
            local_imports.append(extract_node_text(node).strip())

    # 2. Re-process full file
    for node in root.children:
        if node.type in {"import_statement", "import_from_statement", "future_import_statement"}:
            continue

        if node.type == "class_definition":
            class_code = extract_node_text(node)
            class_name = extract_class_name(node)
            class_description = summarize_class_with_qwen(class_code)

            # Collect all method names
            method_nodes = [
                child for child in node.child_by_field_name("body").children
                if child.type == "function_definition"
            ]
            method_names = []
            for m in method_nodes:
                name_node = m.child_by_field_name("name")
                if name_node:
                    method_names.append(extract_node_text(name_node))

            # For each method, build chunk with description + other methods listed
            for method_node in method_nodes:
                this_method_name = extract_node_text(method_node.child_by_field_name("name"))
                other_methods = [n for n in method_names if n != this_method_name]
                method_decorators = extract_decorators_above(method_node)
                method_text = extract_node_text(method_node).rstrip()

                comment_lines = [
                    f"# Method from class '{class_name}': {class_description}",
                    f"# Other methods in class: {', '.join(other_methods) if other_methods else 'None'}"
                ]

                full_chunk = "\n".join(
                    comment_lines + method_decorators + [method_text]
                )
                chunks.append(full_chunk)

        elif node.type == "function_definition":
            method_decorators = extract_decorators_above(node)
            full_text = extract_node_text(node).rstrip()
            chunk = "\n".join(method_decorators + [full_text])
            chunks.append(chunk)

        else:
            node_text = extract_node_text(node).strip()
            if node_text:
                full_chunk = "\n".join(local_imports + [node_text])
                chunks.append(node_text)

    return chunks

