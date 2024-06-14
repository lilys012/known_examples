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

data_name = sys.argv[1]
if data_name not in ['ambiqa', 'quest', 'qampari']: raise Exception('invalid data_name')
strategy = sys.argv[2]
if strategy not in ['none', 'some', 'full', 'base']: raise Exception('invalid strategy')
split = sys.argv[3]
if (data_name == 'ambiqa' and split != 'dev') or split not in ['dev', 'test']: raise Exception('invalid split')
port = sys.argv[4]

split_dataset_f = open("../jsons/"+data_name+"/"+split+"_dataset.json")
split_dataset = json.load(split_dataset_f)
split_dataset_f.close()


answer_sets_f = open("../jsons/"+data_name+"_experiments/answer_set/answer_sets.json")
answer_sets = json.load(answer_sets_f)
answer_sets_f.close()

# retrieve target prompt examples
answer_set = []
for i in range(2 if data_name == "quest" and strategy in ["some", "full"] else 4):
    answer_set.append(answer_sets[strategy+"_pk_set"+str(i+1)])

def example_to_string(example):
    string_rep = "Question: " + example["question"] + "\nAnswers: "
    for i, answer in enumerate(random.sample(example["answers"], len(example["answers"]))):
        string_rep += answer
        if i != len(example["answers"])-1:
            string_rep += " | "
    return string_rep

for set_idx in range(len(answer_set)):
    answer_set_strings = []
    for example in answer_set[set_idx]:
        answer_set_strings.append(example_to_string(example))

    experiment_log = []
    questions_left_unanswered = 0
    errors = []

    for query_idx in tqdm(range(len(split_dataset)), desc="Set"+str(set_idx+1)):
        example_log = {}

        split_example = split_dataset[query_idx]

        example_log[split+"_example_idx"] = query_idx
        example_log[split+"_example"] = split_example

        prompt = ""
        for prompt_example in answer_set_strings:
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
            errors.append(query_idx)
        example_log["output_text"] = output_text

        experiment_log.append(example_log)

    save_path = "../jsons/"+data_name+"_experiments/answer_set/"+strategy
    os.makedirs(save_path, exist_ok=True)
    f = open(save_path+"/"+split+"_answer_set_"+str(set_idx+1)".json", "w")
    json.dump(experiment_log, f, indent=4)
    f.close()

    print("Errors: ")
    print(errors)



    