from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_kotlin as ts_kotlin

# Load file content
file_path = "data/repositories-python-practice/celery__kombu-7ccec0b5369f94c51bdf487ac274a68c4b9bdfb9/kombu/tests/transport/test_redis.py"
with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Load language
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


code_bytes = code.encode("utf-8")
tree = parser.parse(code_bytes)
root = tree.root_node

def extract_node_text(code_bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

def extract_with_decorators(code_bytes, node):
    decorators = []
    prev = node.prev_sibling
    while prev and prev.type == "decorator":
        decorators.insert(0, extract_node_text(code_bytes, prev).rstrip())
        prev = prev.prev_sibling
    node_text = extract_node_text(code_bytes, node).rstrip()
    return "\n".join(decorators + [node_text])

def extract_class_info(decorated_def_node):
    class_def = next((c for c in decorated_def_node.children if c.type == "class_definition"), None)
    if not class_def:
        return

    class_name_node = class_def.child_by_field_name("name")
    class_name = extract_node_text(code_bytes, class_name_node)

    decorators = [extract_node_text(code_bytes, d).rstrip()
                  for d in decorated_def_node.children if d.type == "decorator"]

    print(f"\nClass: {class_name}")
    if decorators:
        print(f"  Decorators: {decorators}")

    body = class_def.child_by_field_name("body")
    if body:
        for item in body.children:
            if item.type == "function_definition":
                method_name_node = item.child_by_field_name("name")
                method_name = extract_node_text(code_bytes, method_name_node)
                method_decorators = [
                    extract_node_text(code_bytes, d).rstrip()
                    for d in item.children if d.type == "decorator"
                ]
                print(f"  Method: {method_name}")
                print(f"Method content: {extract_node_text(code_bytes, item)}")
                if method_decorators:
                    print(f"    Decorators: {method_decorators}")

# Walk through the top-level nodes
for node in root.children:
    if node.type == "decorated_definition":
        extract_class_info(node)
    elif node.type == "class_definition":
        # Handle undecorated classes
        dummy_wrapper = node  # Fake a decorated_definition for uniformity
        extract_class_info(dummy_wrapper)
