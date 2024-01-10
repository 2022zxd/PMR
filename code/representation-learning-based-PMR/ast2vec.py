# compile the Python code into a syntax tree
#with open("add_two_array_values.py", "r") as f:
    #content = f.read()
#src = content
import ast
import python_ast_utils
import torch
# load the ast2vec model
import ast2vec
model = ast2vec.load_model()
# translate to a vector

encoded_list = []
for i in range(1, 98):
    file_path = f"python/python_{i}.py"
    with open(file_path, "r") as f:
        src = f.read()
    tree = python_ast_utils.ast_to_tree(ast.parse(src))
    print(tree)
    with torch.no_grad():
        x = model.encode(tree)
    encoded_list.append(x.cpu().numpy())
    print(i)
    print(encoded_list)
import numpy as np
import pandas as pd
df = pd.DataFrame(np.array(encoded_list))
df.to_csv('ast2vec-python.csv', index=False, header=False)