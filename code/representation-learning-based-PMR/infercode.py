from infercode.client.infercode_client import InferCodeClient
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

# Change from -1 to 0 to enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# cpp
infercode = InferCodeClient(language="java")
infercode.init_from_config()
vectors = []
for i in range(1, 101):
    filename = f"d:/Users/Administrator/Desktop/infercode-master/java/java_{i}.java"
    with open(filename, "r") as f:
        code = f.read()
    print(code)
    vector = infercode.encode([code])[0]
    vectors.append(vector)

df = pd.DataFrame(vectors)
df.to_csv("infercode-java-1.csv", index=False)

print(vectors)

