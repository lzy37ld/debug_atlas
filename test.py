# from nomic import atlas
# import numpy as np

# num_embeddings = 1000
# embeddings = np.random.rand(num_embeddings, 10)
# data = [{'upload': '1', 'id': i} for i in range(len(embeddings))]

# project = atlas.map_embeddings(embeddings=embeddings,
#                                data=data,
#                                id_field='id',
#                                name='A Map That Gets Updated',
#                                colorable_fields=['upload'],
#                                reset_project_if_exists=True)

# map = project.get_map('A Map That Gets Updated')
# print(1)
# print(map)


from nomic import atlas
import nomic
import numpy as np
import random
from datasets import load_dataset



sharegpt_path = "./ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
import json
with open(sharegpt_path) as f:
    sharegpt_data = json.load(f)

extracted_sharegpt_data = []
for _sharegpt_data in sharegpt_data:
    try:
        if _sharegpt_data["conversations"][0]["from"] == "human" and _sharegpt_data["conversations"][0]["value"].strip() != "":
            extracted_sharegpt_data.append(_sharegpt_data["conversations"][0]["value"])
    except:
        pass
datasets_raw = extracted_sharegpt_data
datasets_raw = list(set(datasets_raw))
assert "" not in datasets_raw, "error here"
for i in datasets_raw:
    assert type(i) == str, "false"

dataset = [{"from":"human", "value":_} for _ in datasets_raw]
indexed_field = "value"

max_documents = 5000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

project = atlas.map_text(data=documents,
                         indexed_field = indexed_field,
                         name='sharegpt_lzy',
                         reset_project_if_exists=True)
