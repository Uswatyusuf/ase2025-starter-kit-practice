from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin


PY_LANGUAGE = Language(tspython.language())
KT_LANGUAGE = Language (ts_kotlin.language())

def chunk_python(code: str) -> list[str]:
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
        if node.type in {"future_import_statement", "import_statement", "import_from_statement", "package_header", "import_list", "import_header", "import_statement"}:
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




def chunk_kotlin_method_lvl(code: str) -> list[str]:
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

