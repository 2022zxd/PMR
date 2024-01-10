import csv

from unixcoder import UniXcoder
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/unixcoder-base"
model = UniXcoder(model_name)
# model.to(device)
encoded_list = []
for i in range(1, 98):
    file_path = f"/home/zxd/Downloads/CodeBERT-master/python/python_{i}.py"
    with open(file_path, "r") as f:
        code = f.read()
    print(code)
    tokens_ids = model.tokenize([code], max_length=128,mode="<encoder-only>")
    print(tokens_ids)
    # source_ids = torch.tensor(tokens_ids).to(device)
    source_ids = torch.tensor(tokens_ids)
    print(source_ids)
    tokens_embeddings, func_embedding = model(source_ids)
    print(i)
    print(func_embedding)
    with open('UniXcoder-python.csv',mode='a',newline='') as file:
        writter = csv.writer(file)
        writter.writerow(func_embedding.detach().cpu().numpy()[0])