import os
import jsonlines
from tree_sitter import Language, Parser
import tree_sitter_kotlin as tskotlin

KT_LANGUAGE = Language(tskotlin.language())
parser = Parser(KT_LANGUAGE)


def find_actual_block(node):
    # Recursively find the first function/class/object declaration inside any node
    if node.type in ("function_declaration", "class_declaration", "object_declaration"):
        return node
    for child in node.children:
        found = find_actual_block(child)
        if found:
            return found
    return None


def extract_prefix_from_last_block_kt(code_bytes, caret_byte_offset):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    best_node = None
    for node in root.children:
        if not node.is_named:
            continue
        actual = find_actual_block(node)
        if actual and actual.start_byte <= caret_byte_offset:
            if best_node is None or actual.start_byte > best_node.start_byte:
                best_node = actual

    if best_node:
        return code_bytes[best_node.start_byte:caret_byte_offset].decode("utf-8", errors="ignore")
    else:
        return code_bytes[:caret_byte_offset].decode("utf-8", errors="ignore")


def extract_suffix_to_next_block_kt(code_bytes, caret_byte_offset):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    next_block_start = len(code_bytes)
    for node in root.children:
        if not node.is_named:
            continue
        if node.start_byte > caret_byte_offset and node.type in (
            "function_declaration", "class_declaration", "object_declaration"
        ):
            next_block_start = node.start_byte
            break

    suffix = code_bytes[caret_byte_offset:next_block_start]
    return suffix.decode("utf-8", errors="ignore")


def debug_print_top_level_nodes(code_bytes):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    print("\n🔎 Top-level nodes:")
    for node in root.children:
        if not node.is_named:
            continue
        snippet = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        first_line = snippet.strip().split("\n")[0]
        print(f"- {node.type} @ byte {node.start_byte}: {first_line}")


