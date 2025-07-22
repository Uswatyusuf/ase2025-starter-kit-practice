import ast

class ClassMethodExtractor(ast.NodeVisitor):
    def __init__(self):
        self.class_methods = {}

    def visit_ClassDef(self, node):
        method_names = [
            n.name for n in node.body
            if isinstance(n, ast.FunctionDef)
        ]
        self.class_methods[node.name] = method_names
        self.generic_visit(node)

# Usage
with open("data/repositories-python-practice/celery__kombu-0d3b1e254f9178828f62b7b84f0307882e28e2a0/t/unit/transport/test_redis.py", 'r') as f:
    tree = ast.parse(f.read())

extractor = ClassMethodExtractor()
extractor.visit(tree)

for cls, methods in extractor.class_methods.items():
    print(f"Class: {cls}")
    for method in methods:
        print(f"  Method: {method}")
