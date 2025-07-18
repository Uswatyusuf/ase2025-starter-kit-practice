from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin
from transformers import AutoTokenizer

file_path = "data/repositories-python-practice/celery__kombu-7f9674419b585921b1da4ecbd5f3dc203891955e/kombu/utils/eventio.py"
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
MAX_TOKENS = 512
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


# def basic_ast_chunk_code(code: str, lang: str) -> list[str]:
#     global parser
#     print("Language; ", lang)
#     if lang == "python":
#         parser = Parser(PY_LANGUAGE)
#     elif lang == "kotlin":
#         parser = Parser(KT_LANGUAGE)
#     code_bytes = code.encode("utf-8")
#     tree = parser.parse(code_bytes)
#     root = tree.root_node
#
#     chunks = []
#     current_expr_group = []
#
#     def flush_expr_group():
#         if current_expr_group:
#             chunk_text = "\n".join(
#                 code_bytes[n.start_byte:n.end_byte].decode("utf-8").strip()
#                 for n in current_expr_group
#             )
#             if chunk_text.strip():
#                 chunks.append(chunk_text)
#             current_expr_group.clear()
#
#     for node in root.children:
#         # Skip import statements
#         if node.type in {"future_import_statement", "import_statement", "import_from_statement"}:
#             continue
#
#         if node.type == "expression_statement":
#             current_expr_group.append(node)
#         else:
#             flush_expr_group()
#             chunk_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
#             if chunk_text.strip():
#                 chunks.append(chunk_text)
#
#     flush_expr_group()
#     return chunks

# def clean_chunk_text(text: str) -> str:
#     lines = [line.rstrip() for line in text.splitlines()]
#     cleaned = "\n".join(line for line in lines if line.strip())
#     return cleaned.strip()
#
# def ast_chunk_code_with_comments(code: str, lang: str) -> list[str]:
#     global parser
#     if lang == "python":
#         parser = Parser(PY_LANGUAGE)
#     elif lang == "kotlin":
#         parser = Parser(KT_LANGUAGE)
#     else:
#         raise ValueError(f"Unsupported language: {lang}")
#
#     code_bytes = code.encode("utf-8")
#     tree = parser.parse(code_bytes)
#     root = tree.root_node
#
#     chunks = []
#     current_expr_group = []
#
#     def flush_expr_group():
#         if current_expr_group:
#             chunk_text = "\n".join(
#                 clean_chunk_text(code_bytes[n.start_byte:n.end_byte].decode("utf-8"))
#                 for n in current_expr_group
#             )
#             if chunk_text:
#                 header = "# Expression Group\n"
#                 chunks.append(header + chunk_text)
#             current_expr_group.clear()
#
#     for node in root.children:
#         node_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
#
#         if node.type in {"future_import_statement", "import_statement", "import_from_statement"}:
#             continue
#
#         if node.type == "expression_statement":
#             current_expr_group.append(node)
#         else:
#             flush_expr_group()
#
#             # Detect function or class name
#             name = None
#             for child in node.children:
#                 if child.type == "identifier":
#                     name = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
#                     break
#
#             if node.type == "function_definition":
#                 header = f"# Function: {name}\n"
#             elif node.type == "class_definition":
#                 header = f"# Class: {name}\n"
#             else:
#                 header = f"# {node.type.replace('_', ' ').capitalize()}\n"
#
#             chunk_cleaned = clean_chunk_text(node_text)
#             chunks.append(header + chunk_cleaned)
#
#     flush_expr_group()
#     return chunks
#
# def print_ast_structure(code: str):
#     PY_LANGUAGE = Language(tspython.language())
#     parser = Parser(PY_LANGUAGE)
#     code_bytes = code.encode("utf-8")
#     tree = parser.parse(code_bytes)
#     root = tree.root_node
#
#     print("Root type:", root.type)
#
#     for i, node in enumerate(root.children):
#         node_type = node.type
#         node_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8").strip()
#         print(f"--- Node {i+1} ---")
#         print(f"Type: {node_type}")
#         print(f"Text:\n{node_text}")
#         print()


def token_count(text: str) -> int:
    return len(tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0])

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
    current_token_count = token_count(current_chunk)

    for method in methods:
        method_text = extract_node_text(code_bytes, method)
        method_text_clean = clean_chunk_text(method_text)
        method_tokens = token_count(method_text_clean)

        if current_token_count + method_tokens > max_tokens:
            # Flush current chunk
            chunks.append(current_chunk.rstrip())
            # Start new chunk with header + method
            current_chunk = header + method_text_clean + "\n"
            current_token_count = token_count(current_chunk)
        else:
            current_chunk += method_text_clean + "\n"
            current_token_count += method_tokens

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
#chunk code with context
def ast_chunk_code_with_context(code: str, lang: str) -> list[str]:
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

    def clean_chunk_text(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line.strip()).strip()

    def get_name(node):
        for child in node.children:
            if child.type == "identifier":
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def get_docstring(node):
        for child in node.children:
            if child.type == "expression_statement":
                text = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
                if text.strip().startswith(("'''", '"""')):
                    return text.strip()
        return None

    def extract_class_docstrings():
        class_docs = {}
        def recurse(node):
            if node.type == "class_definition":
                class_name = get_name(node)
                if class_name:
                    doc = get_docstring(node)
                    if doc:
                        class_docs[class_name] = doc
            for child in node.children:
                recurse(child)
        recurse(root)
        return class_docs

    class_docs = extract_class_docstrings()

    def get_enclosing_class_name(node):
        parent = node.parent
        while parent:
            if parent.type == "class_definition":
                return get_name(parent)
            parent = parent.parent
        return None

    def process_node(node):
        if node.type == "function_definition":
            name = get_name(node)
            if not name:
                return

            header = f"# Function: {name}"
            enclosing_class = get_enclosing_class_name(node)
            if enclosing_class:
                header += f"\n# Enclosing class: {enclosing_class}"
                if enclosing_class in class_docs:
                    header += f"\n# Class docstring:\n{class_docs[enclosing_class]}"

            docstring = get_docstring(node)
            if docstring:
                header += f"\n# Function docstring:\n{docstring}"

            chunk_body = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            chunk_text = f"{header}\n{clean_chunk_text(chunk_body)}"
            chunks.append(chunk_text)

        # Recurse into all children to find nested functions
        for child in node.children:
            process_node(child)

    # Process top-level children
    for child in root.children:
        process_node(child)

    return chunks
# language = "python"
# chunks = basic_ast_chunk_code(code, language)
# print_ast_structure(code)


# with open("debug_chunks_basic.txt", "w", encoding="utf-8") as f:
#     for i, chunk in enumerate(chunks):
#         f.write(f"--- Chunk {i+1} ---\n")
#         f.write(chunk.strip())
#         f.write("\n\n")
