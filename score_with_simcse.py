import os
import json
import torch
from tqdm import tqdm
from simcse import SimCSE
import sys

data_name = sys.argv[1] # ['ambiqa', 'qampari', 'quest']
if data_name not in ['ambiqa', 'quest', 'qampari']: raise Exception('invalid data_name')

embedding_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

train_dataset_f = open("../jsons/"+data_name+"/train_dataset.json")
train_dataset = json.load(train_dataset_f)
train_dataset_f.close()

train_dataset_reduced_f = open("../jsons/"+data_name+"/train_dataset_reduced.json")
train_dataset_reduced = json.load(train_dataset_reduced_f)
train_dataset_reduced_f.close()

dev_dataset_f = open("../jsons/"+data_name+"/dev_dataset.json")
dev_dataset = json.load(dev_dataset_f)
dev_dataset_f.close()

if data_name != "ambiqa":
    test_dataset_f = open("../jsons/"+data_name+"/test_dataset.json")
    test_dataset = json.load(test_dataset_f)
    test_dataset_f.close()

def example_to_str(example):
    str_rep = "Query: " + example["question"] + "\n" + "Answers: "
    for i, answer in enumerate(example["answers"]):
        str_rep += answer
        if i != len(example["answers"])-1:
            str_rep += " | "
    return str_rep

train_embeddings = []
for example in tqdm(train_dataset):
    example_str = "Query: " + example["question"]
    embedding = embedding_model.encode(example_str)
    train_embeddings.append(embedding)

train_embeddings = torch.stack(train_embeddings)
torch.save(train_embeddings, "jsons/"+data_name+"/simcse_scores/train_embeddings.pt")

train_reduced_embeddings = []
for example in tqdm(train_reduced_dataset):
    example_str = "Query: " + example["question"]
    embedding = embedding_model.encode(example_str)
    train_reduced_embeddings.append(embedding)

train_reduced_embeddings = torch.stack(train_reduced_embeddings)
torch.save(train_reduced_embeddings, "jsons/"+data_name+"/simcse_scores/train_reduced_embeddings.pt")

dev_embeddings = []
for example in tqdm(dev_dataset):
    example_str = "Query: " + example["question"] # We cannot embed the answers (that's cheating!)
    embedding = embedding_model.encode(example_str)
    dev_embeddings.append(embedding)

dev_embeddings = torch.stack(dev_embeddings)
torch.save(dev_embeddings, "jsons/"+data_name+"/simcse_scores/dev_embeddings.pt")

if data_name != "ambiqa":
    test_embeddings = []
    for example in tqdm(test_dataset):
        example_str = "Query: " + example["question"] # We cannot embed the answers (that's cheating!)
        embedding = embedding_model.encode(example_str)
        test_embeddings.append(embedding)

    test_embeddings = torch.stack(test_embeddings)
    torch.save(test_embeddings, "jsons/"+data_name+"/simcse_scores/test_embeddings.pt")