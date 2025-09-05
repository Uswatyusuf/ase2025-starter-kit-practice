from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin
from transformers import AutoTokenizer


mellum_tokenizer = AutoTokenizer.from_pretrained("JetBrains/Mellum-4b-sft-python")





PY_LANGUAGE = Language(tspython.language())
KT_LANGUAGE = Language (ts_kotlin.language())


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
    """
       Split Python source code into chunks using a Tree-sitter AST:
           - Parses the given code into an abstract syntax tree
           - Groups consecutive expression statements together
           - Treats other top-level nodes (functions, classes, etc.) as separate chunks
           - Skips import statements to reduce noise
           - Returns a list of code snippets preserving original formatting

       :param code: Python source code as a string.
       :return: List of code chunks extracted from the AST, each as a string.
       """
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
    """
        Split source code into chunks containing only methods and functions:
            - Parses the given code into an AST using Tree-sitter
            - Preserves top-level imports to include in each extracted chunk
            - Extracts functions and class methods, including decorators
            - For classes, generates a class header and groups decorators with each method
            - Returns a list of method-level code chunks as strings

        :param code: Source code as a string.
        :param lang: Programming language name (currently only "python" is supported).
        :return: List of code chunks, each representing a function or method with relevant context.
        """
    global parser
    if lang.lower() == "python":
        parser = Parser(PY_LANGUAGE)
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

    # Collect top-level imports
    local_imports = []
    for node in root.children:
        if node.type in {"import_statement", "import_from_statement", "future_import_statement"}:
            local_imports.append(extract_node_text(node).strip())

    # Re-process and extract method/function/class chunks
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


def chunk_kotlin_method_lvl_full_context(code: str) -> list[str]:
    """
        Split Kotlin source code into function-level chunks with full context:
            - Parses the code into an AST using Tree-sitter
            - Extracts function declarations along with their annotations and modifiers
            - Collects surrounding class, object, interface, or companion object context
            - Preserves documentation comments (KDoc) above functions or parent declarations
            - Returns function-level chunks enriched with their enclosing context

        :param code: Kotlin source code as a string.
        :return: List of code chunks, each containing a function with annotations and parent context.
        """
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

