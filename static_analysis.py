from tree_sitter import Language,Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin

file_path = "data/repositories-python-practice/celery__kombu-0d3b1e254f9178828f62b7b84f0307882e28e2a0/t/unit/transport/test_redis.py"
file_path_2 = "data/repositories-python-practice/celery__kombu-7ccec0b5369f94c51bdf487ac274a68c4b9bdfb9/kombu/tests/transport/test_redis.py"
# Read the actual file content
with open(file_path_2, 'r', encoding='utf-8') as f:
    code = f.read()

PY_LANGUAGE = Language(tspython.language())
KT_LANGUAGE = Language (ts_kotlin.language())


lang = "Python"
global parser
if lang == "Python":
    parser = Parser(PY_LANGUAGE)
else:
    parser = Parser(KT_LANGUAGE)

code_bytes = code.encode("utf-8")
tree = parser.parse(code_bytes)
root = tree.root_node

def extract_node_text(code_bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

def extract_with_decorators(code_bytes, node):
    """Extract node text including decorators directly attached to it."""
    decorators = []
    idx = node.prev_sibling
    while idx and idx.type == "decorator":
        decorators.insert(0, extract_node_text(code_bytes, idx).rstrip())
        idx = idx.prev_sibling
    node_text = extract_node_text(code_bytes, node).rstrip()
    return "\n".join(decorators + [node_text])

def extract_class_header_methods_only(node, code_bytes):
    """Extract only the class header (without methods)."""
    # From start of class_node to the colon (:), typically first line
    # Use node.start_byte to node.body.start_byte if available
    class_name = ""
    for child in node.children:
        if child.type == "identifier":
            print("identifier is found!")
            class_name = extract_node_text(code_bytes, child)
            break
    return f"class {class_name}:"


def extract_class_decorators(code_bytes, class_node):
    """Extract decorators attached to the class itself."""
    decorators = []
    idx = class_node.prev_sibling
    while idx and idx.type == "decorated_definition":
        decorators.insert(0, extract_node_text(code_bytes, idx).rstrip())
        idx = idx.prev_sibling
    return "\n".join(decorators).rstrip()

for node in  root.children:
    if node.type == "decorated_definition":
        for c in node.children:
            if c.type ==  "class_definition":
                class_decorators = extract_class_decorators(code_bytes, node)
                class_header = extract_class_header_methods_only(c, code_bytes)
                for c_ch in c.children:
                    if c_ch.type == "block":
                        for c_ch_ch in c_ch.children:
                            #print(c_ch_ch)
                            methods = [c_ch_ch_ch for c_ch_ch_ch in c_ch.children if c_ch_ch.type == "function_definition"]
                            print(methods)
                            for method in methods:
                                method_text = extract_with_decorators(code_bytes, method)
                                print(method_text)
                            #     print("Method....")
                            #     print("\n")
                            #     combined = "\n".join(filter(None, [class_decorators, class_header, method_text]))
                            #     print(combined)
                            #     print("\n\n\n")
