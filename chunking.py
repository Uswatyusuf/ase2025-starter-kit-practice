from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin

file_path = "data/repositories-python-practice/celery__kombu-7f9674419b585921b1da4ecbd5f3dc203891955e/kombu/utils/eventio.py"

# Read the actual file content
with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()



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


def basic_ast_chunk_code(code: str, lang: str) -> list[str]:
    global parser
    PY_LANGUAGE = Language(tspython.language())
    KT_LANGUAGE = Language (ts_kotlin.language())
    print("Language; ", lang)
    if lang == "python":
        parser = Parser(PY_LANGUAGE)
    elif lang == "kotlin":
        parser = Parser(KT_LANGUAGE)
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

language = "python"
chunks = basic_ast_chunk_code(code, language)
print_ast_structure(code)


with open("debug_chunks_basic.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- Chunk {i+1} ---\n")
        f.write(chunk.strip())
        f.write("\n\n")
