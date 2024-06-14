import os
import json
import torch
import torch.nn as nn 
import numpy as np
import sys
from tqdm import tqdm 
import random
import requests
import time
from copy import deepcopy
from functools import cmp_to_key

data_name = sys.argv[1] # ['ambiqa', 'qampari', 'quest']
if data_name not in ['ambiqa', 'quest', 'qampari']: raise Exception('invalid data_name')
port = sys.argv[2]
split = sys.argv[3]
order = sys.arg[4]
if order not in ['grd', 'rev', 'asc', 'desc', 'rand', 'alpha']: raise Exception('invalid ordering strategy')

save_path = "../jsons/"+data_name+"_experiments/answer_ordering/"+split+"_"+order+".json"

split_dataset_f = open("../jsons/"+data_name+"/"+split+"_dataset.json")
split_dataset = json.load(split_dataset_f)
split_dataset_f.close()

if order in ['grd', 'rev']:
    train_dataset_f = open("../jsons/"+data_name+"_experiments/answer_ordering/train_dataset_"+("reverse_" if order=='rev' else "")+"greedy.json")
elif order in ['asc', 'desc']:
    train_dataset_f = open("../jsons/"+data_name+"_experiments/answer_ordering/train_dataset_"+order+"end.json")
else:
    train_dataset_f = open("../jsons/"+data_name+"/train_dataset_reduced.json")
train_dataset = json.load(train_dataset_f)
train_dataset_f.close()

def example_to_string(example):
    string_rep = "Question: " + example["question"] + "\nAnswers: "
    for i, answer in enumerate(sorted(example["answers"]) if order=='alpha' else random.sample(example["answers"], len(example["answers"])) if order=='rand' else example["answers"]):
        string_rep += answer
        if i != len(example["answers"])-1:
            string_rep += " | "
    return string_rep

train_example_strings = []
for train_ex in train_dataset:
    train_example_strings.append(example_to_string(train_ex))

experiment_log = []
questions_left_unanswered = 0

split_embeddings = torch.load("../jsons/"+data_name+"/simcse_scores/"+split+"_embeddings.pt")
train_embeddings = torch.load("../jsons/"+data_name+"/simcse_scores/train_reduced_embeddings.pt")
split_train_sim_matrix = torch.matmul(split_embeddings, torch.transpose(train_embeddings, 0, 1)).cpu().numpy()

errors = []
for query_idx in tqdm(range(len(split_dataset))):
    if True:
        example_log = {}

        split_example = split_dataset[query_idx]

        example_log[split+"_example_idx"] = query_idx
        example_log[split+"_example"] = split_example

        sim_scores = split_train_sim_matrix[query_idx]
        sorted_indices = [i for i in range(len(sim_scores))]
        sorted_indices = sorted(sorted_indices, key = lambda x: -sim_scores[x])

        prompt_example_indices_in_train = deepcopy(sorted_indices[:5])
        prompt_example_indices_in_train.reverse()

        example_log["prompt_example_indices_in_train"] = prompt_example_indices_in_train

        prompt_examples = [train_example_strings[sorted_indices[i]] for i in range(5)]
        prompt_examples.reverse() # So that the most similar example is last

        prompt = ""
        for prompt_example in prompt_examples:
            prompt += prompt_example + "\n\n"
        prompt += "Question: " + split_example["question"] + "\nAnswers:"

        example_log["prompt"] = prompt

        max_new_tokens = 30 if data_name == "ambiqa" else 150 if data_name == "quest" else 100
        kwargs = {"prompts" : [prompt], "min_new_tokens" : 2, "max_new_tokens" : max_new_tokens, "num_beams" : 5, "eos_token_id" : 13}
        kwargs_str = json.dumps(kwargs)
        params = {"kwargs" : kwargs_str}
        response = requests.get("http://127.0.0.1:"+port+"/", params=params)
        try:
            response = json.loads(response.text)
            output_text = response[0]["output_text"]
        except:
            output_text = ""
            questions_left_unanswered += 1
            errors.append(query_idx)
        example_log["output_text"] = output_text

        experiment_log.append(example_log)

f = open(save_path, "w")
json.dump(experiment_log, f, indent=4)
f.close()

print("Num questions left unanswered:", questions_left_unanswered)
print(errors)



    