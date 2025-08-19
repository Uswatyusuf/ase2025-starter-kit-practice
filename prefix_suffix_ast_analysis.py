import os
import json
import jsonlines
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import matplotlib.pyplot as plt
from collections import Counter



# Load Tree-sitter Python grammar
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extract_prefix_from_last_block(code_bytes, caret_byte_offset):
    last_block = find_last_top_level_block_before(code_bytes, caret_byte_offset)

    if last_block:
        return code_bytes[last_block.start_byte:caret_byte_offset].decode("utf-8", errors="ignore")
    else:
        # fallback: return all code up to caret
        return code_bytes[:caret_byte_offset].decode("utf-8", errors="ignore")



def extract_suffix_to_next_block(code_bytes, caret_byte_offset):
    tree = parser.parse(code_bytes)
    root = tree.root_node

    next_block_start = len(code_bytes)  # default: end of file

    # Debug print: all top-level nodes *after* caret
    # print("\n🔎 Top-level nodes after caret:")
    for node in root.children:
        if not node.is_named:
            continue
        if node.start_byte > caret_byte_offset and node.type in (
            "function_definition", "class_definition", "decorated_definition"
        ):
           # print(f"- {node.type} starting at byte {node.start_byte}:")
            code_snippet = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            # print(code_snippet.strip().split("\n")[0] + " ...")
            next_block_start = node.start_byte
            break  # stop at the first valid block

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

# Set up input path
language = "python"
stage = "public"
completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
gt_file = os.path.join("answers", f"answers-{language}-{stage}.jsonl")

# # Process examples
# with jsonlines.open(completion_points_file, 'r') as reader:
#     for instance_id, datapoint in enumerate(reader):
#         prefix = datapoint["prefix"]
#         suffix = datapoint["suffix"]
#         code_bytes = (prefix + suffix).encode("utf-8")
#         px_code_bytes = prefix.encode("utf-8")
#         caret_offset = len(prefix.encode("utf-8"))
#
#         extracted_prefix = extract_prefix_from_last_block(px_code_bytes, caret_offset)
#         extracted_suffix = extract_suffix_to_next_block(code_bytes, caret_offset)
#
#         print(f"💡 Extracted prefix {instance_id} (from last top-level block):")
#         print(extracted_prefix)
#         print("💡 Extracted suffix (to next top-level block):")
#         print(extracted_suffix)
#         print("=" * 60)
#
#
# # Track extracted block types
# block_type_counter = Counter()
#
#
#
# decorated_subtype_counter = Counter()  # Track breakdown inside decorated_definition
#
# with jsonlines.open(completion_points_file, 'r') as reader:
#     for instance_id, datapoint in enumerate(reader):
#         prefix = datapoint["prefix"]
#         code_bytes = prefix.encode("utf-8")
#         caret_offset = len(code_bytes)
#         closest_node = find_last_top_level_block_before(code_bytes, caret_offset)
#
#         if closest_node:
#             node_type = closest_node.type
#
#             if node_type == "decorated_definition":
#                 # Determine if it's decorating a function or class
#                 decorated_child = next(
#                     (child for child in closest_node.children if child.type in ("function_definition", "class_definition")),
#                     None
#                 )
#                 if decorated_child:
#                     if decorated_child.type == "function_definition":
#                         decorated_subtype_counter["decorated_function"] += 1
#                     elif decorated_child.type == "class_definition":
#                         decorated_subtype_counter["decorated_class"] += 1
#
#             block_type_counter[node_type] += 1
#
# # --- Plotting ---
#
# labels = ["function_definition", "class_definition", "decorated_definition"]
# counts = [block_type_counter["function_definition"],
#           block_type_counter["class_definition"],
#           block_type_counter["decorated_definition"]]
#
# # For decorated_definition, split into function/class
# decorated_func = decorated_subtype_counter["decorated_function"]
# decorated_class = decorated_subtype_counter["decorated_class"]
#
# # Base bars
# bar_width = 0.5
# fig, ax = plt.subplots()
#
# # Plot individual bars
# bars = ax.bar(labels, counts, width=bar_width, color=["skyblue", "lightgreen", "lightgray"])
#
# # Stacked sub-bar for decorated_definition
# # Add the split if decorated_definition exists
# if block_type_counter["decorated_definition"]:
#     ax.bar("decorated_definition", decorated_func, width=bar_width, color="orange", label="decorated_function")
#     ax.bar("decorated_definition", decorated_class, width=bar_width, bottom=decorated_func, color="purple", label="decorated_class")
#
# # Labels and legend
# ax.set_ylabel("Count")
# ax.set_title("Top-level Blocks Extracted")
# ax.legend()
# plt.tight_layout()
# plt.show()


# saving extractions to files
# Output folder
output_dir = os.path.join("outputs", f"{language}-{stage}")
os.makedirs(output_dir, exist_ok=True)

# Load ground truth answers into memory
with jsonlines.open(gt_file, 'r') as gt_reader:
    ground_truths = list(gt_reader)

# Process and save to .py files
with jsonlines.open(completion_points_file, 'r') as reader:
    for instance_id, datapoint in enumerate(reader):
        prefix = datapoint["prefix"]
        suffix = datapoint["suffix"]
        code_bytes = (prefix + suffix).encode("utf-8")
        px_code_bytes = prefix.encode("utf-8")
        caret_offset = len(px_code_bytes)

        extracted_prefix = extract_prefix_from_last_block(px_code_bytes, caret_offset)
        extracted_suffix = extract_suffix_to_next_block(code_bytes, caret_offset)

        ground_truth_middle = ground_truths[instance_id]["middle"]

        file_content = f'''# === PREFIX ===
{extracted_prefix.strip()}

# === MIDDLE ===
{ground_truth_middle.strip()}

# === SUFFIX ===
{extracted_suffix.strip()}
'''

        file_name = f"{instance_id:04d}.py"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as py_file:
            py_file.write(file_content)

        print(f"✅ Saved {file_path}")