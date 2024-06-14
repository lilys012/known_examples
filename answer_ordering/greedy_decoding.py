import os
import json
import torch
import requests
from simcse import SimCSE
import sys
import tqdm

data_name = sys.argv[1] 
if data_name not in ['ambiqa', 'quest', 'qampari']: raise Exception('invalid data_name')
port = sys.argv[2]
reverse = sys.argv[3] 
if reverse not in ['grd', 'rev']: raise Exception('invalid strategy')

embedding_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

train_dataset_f = open("../jsons/"+data_name+"/train_dataset_reduced.json")
train_dataset = json.load(train_dataset_f)
train_dataset_f.close()

train_dataset_embeddings = torch.load("../jsons/"+data_name+"/simcse_scores/train_reduced_embeddings.pt")

def example_to_string(example):
    string_rep = "Question: " + example["question"] + "\nAnswers: "
    for i, answer in enumerate(example["answers"]):
        string_rep += answer
        if i != len(example["answers"])-1:
            string_rep += " | "
    return string_rep

errs = []
for train_idx, train_ex in enumerate(tqdm.tqdm(train_dataset)):
    # Find the most similar 5 examples to prompt with
    train_ex_embedding = train_dataset_embeddings[train_idx]
    sim_scores = torch.matmul(train_dataset_embeddings, train_ex_embedding)
    sim_scores[train_idx] = -1
    indices_and_scores = [(i, sim_scores[i].item()) for i in range(len(sim_scores))]
    indices_and_scores = sorted(indices_and_scores, key=lambda x: x[1])
    top_5_example_indices = [indices_and_scores[i][0] for i in range(len(indices_and_scores)-5, len(indices_and_scores))]

    prompt = ""
    for idx in top_5_example_indices:
        prompt += example_to_string(train_dataset[idx])
        prompt += "\n\n"
    prompt += "Question: " + train_ex["question"] + "\nAnswers:"
    kwargs = {"prompts" : [prompt], "answer_ordering" : {"type" : ("reverse_" if reverse=='rev' else "")+"greedy_decoding", "answer_set" : train_ex["answers"]}}
    kwargs_str = json.dumps(kwargs)
    params = {"kwargs" : kwargs_str}
    response = requests.get("http://127.0.0.1:"+port+"/", params=params)
    try:
        res_dict = json.loads(response.text)
        greedy_ordering = res_dict["text"]
    except: 
        greedy_ordering = ""
        errs.append(train_idx)
    greedy_ordering = greedy_ordering.split("|")
    greedy_ordering = [greedy_ordering[i].strip() for i in range(len(greedy_ordering))]
    train_ex["answers"] = greedy_ordering
    train_dataset[train_idx] = train_ex

print(errs)

# Save new train example set and new embeddings
train_dataset_f = open("../jsons/"+data_name+"_experiments/answer_ordering/train_dataset_"+("reverse_" if reverse=='rev' else "")+"greedy.json", "w")
json.dump(train_dataset, train_dataset_f, indent=4)
train_dataset_f.close()