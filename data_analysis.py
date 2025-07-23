import os
import jsonlines
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Load Tree-sitter Python grammar
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


# Node types for fallback
COMPOUND_BLOCKS = {
    "if_statement", "for_statement", "while_statement", "try_statement", "with_statement", "except_clause"
}

def find_closest_code_block(code_bytes, caret_byte_offset):
    tree = parser.parse(code_bytes)
    root = tree.root_node
    node = root.named_descendant_for_byte_range(caret_byte_offset - 1, caret_byte_offset - 1)

    compound_block = None
    top_level_stmt = None

    while node and node.type != "module":
        if node.type == "function_definition":
            return node  # Best match
        elif node.type in COMPOUND_BLOCKS:
            if compound_block is None:
                compound_block = node
        elif node.parent and node.parent.type == "module" and top_level_stmt is None:
            top_level_stmt = node
        node = node.parent

    # Fallbacks
    return compound_block or top_level_stmt

# Set up input path
language = "python"
stage = "practice"
completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")

# Process examples
with jsonlines.open(completion_points_file, 'r') as reader:
    for instance_id, datapoint in enumerate(reader):
        prefix = datapoint["prefix"]
        code_bytes = prefix.encode("utf-8")
        caret_offset = len(code_bytes)

        closest_node = find_closest_code_block(code_bytes, caret_offset)

        print(f"\n🔎 Instance {instance_id}")
        if closest_node:
            block_code = code_bytes[closest_node.start_byte:closest_node.end_byte].decode("utf-8", errors="ignore")
            print(f"Closest node type: {closest_node.type}")
            print("---- Extracted Code ----")
            print(block_code)
        else:
            print("❌ No meaningful block found.")
