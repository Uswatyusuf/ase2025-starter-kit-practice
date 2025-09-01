from tree_sitter import Language, Parser
import tree_sitter_python as tspython


# Load Tree-sitter Python grammar
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extract_prefix_from_last_block(code_bytes, caret_byte_offset):
    last_block = find_last_top_level_block_before(code_bytes, caret_byte_offset)

    if last_block:
        return code_bytes[last_block.start_byte:caret_byte_offset].decode("utf-8", errors="ignore")
    else:
        return code_bytes[:caret_byte_offset].decode("utf-8", errors="ignore")


def extract_suffix_to_next_block(code_bytes, caret_byte_offset):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    next_block_start = len(code_bytes)

    # Debug print: all top-level nodes *after* caret
    for node in root.children:
        if not node.is_named:
            continue
        if node.start_byte > caret_byte_offset and node.type in (
            "function_definition", "class_definition", "decorated_definition"
        ):
            next_block_start = node.start_byte
            break

    suffix = code_bytes[caret_byte_offset:next_block_start]
    return suffix.decode("utf-8", errors="ignore")


def find_last_top_level_block_before(code_bytes, caret_byte_offset):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    best_node = None

    for node in root.children:
        if not node.is_named:
            continue
        if node.type in ("function_definition", "class_definition", "decorated_definition"):
            if node.start_byte <= caret_byte_offset:
                if best_node is None or node.start_byte > best_node.start_byte:
                    best_node = node
    return best_node

# Print function definitions for debugging
def debug_print_functions(code_bytes):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    print("\n🌲 Tree-sitter function definitions found:")
    def walk(node):
        if node.type in ("function_definition", "class_definition", "decorated_definition"):
            fn_code = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            print(f"- Starts at byte {node.start_byte}:")
            print(fn_code.strip().split("\n")[0] + " ...")
        for child in node.children:
            walk(child)
    walk(root)

