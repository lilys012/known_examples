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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_train_against_train(data_name):
    data = {}
    name_dict = {"ambiqa":"AmbigQA", "qampari":"QAMPARI", "quest":"QUEST"}
    for data_name in ['ambiqa', 'qampari', 'quest']:
        train_dataset_embeddings = torch.load("../jsons/"+data_name+"/simcse_scores/train_embeddings.pt")
        train_to_train_sim = torch.matmul(train_dataset_embeddings, torch.transpose(train_dataset_embeddings, 0, 1))
        train_to_train_sim_mask = 1-torch.eye(train_to_train_sim.shape[0])
        train_to_train_sim = train_to_train_sim * train_to_train_sim_mask
        train_to_train_sim = torch.sum(train_to_train_sim, dim=1)
        train_to_train_sim = train_to_train_sim / (len(train_to_train_sim)-1)
        data[name_dict[data_name]] = train_to_train_sim.numpy()

        # Filtering by equal number
        example_indices = []
        sorted_train_to_train_sim = sorted(enumerate(train_to_train_sim), key=lambda x: x[1])
        mid = int(len(sorted_train_to_train_sim)/2)
        example_indices.append(mid)
        for i in range(1, 250 if data_name == "quest" else 500):
            example_indices.append(sorted_train_to_train_sim[mid+i][0])
            example_indices.append(sorted_train_to_train_sim[mid-i][0])
        print(sorted_train_to_train_sim[mid-i][1], sorted_train_to_train_sim[mid][1], sorted_train_to_train_sim[mid+i][1], len(example_indices))

    with open("../jsons/"+data_name+"/decently_similar_train_subset.json", "w") as f:
        json.dump(example_indices, f)

def example_to_string(example):
    string_rep = "Question: " + example["question"] + "\nAnswers: "
    for i, answer in enumerate(random.sample(example["answers"], len(example["answers"]))):
        string_rep += answer
        if i != len(example["answers"])-1:
            string_rep += " | "
    return string_rep

def score_generation_output(pred_answers, gt_answers):
    score = {
        "EM" : {
            "F1" : 0,
            "P" : 0,
            "R" : 0
        }
    }

    assert not (len(pred_answers) == 0 or len(gt_answers) == 0)

    #### Set level precision
    for pred_answer in pred_answers:
        for gt_answer in gt_answers:
            if pred_answer == gt_answer:
                score["EM"]["P"] += 1
    score["EM"]["P"] /= len(pred_answers)

    #### Set level recall
    for gt_answer in gt_answers:
        for pred_answer in pred_answers:
            if pred_answer == gt_answer:
                score["EM"]["R"] += 1
    score["EM"]["R"] /= len(gt_answers)

    if score["EM"]["P"] + score["EM"]["R"] != 0:
        score["EM"]["F1"] = 2 * score["EM"]["P"] * score["EM"]["R"] / (score["EM"]["P"] + score["EM"]["R"])
    return score

def eval_on_train_subset(data_name, port):
    train_dataset_f = open(".../jsons/"+data_name+"/train_dataset.json")
    train_dataset = json.load(train_dataset_f)
    train_dataset_f.close()

    subset_indices_f = open("../jsons/"+data_name+"/decently_similar_train_subset.json")
    subset_indices = json.load(subset_indices_f)
    subset_indices_f.close()

    train_dataset_embeddings = torch.load("../jsons/"+data_name+"/simcse_scores/train_embeddings.pt")
    train_to_train_sim = torch.matmul(train_dataset_embeddings, torch.transpose(train_dataset_embeddings, 0, 1))
    train_to_train_sim_mask = 1-torch.eye(train_to_train_sim.shape[0])
    train_to_train_sim = train_to_train_sim * train_to_train_sim_mask
    train_to_train_sim = train_to_train_sim + (1-train_to_train_sim_mask) * -1

    train_example_strings = []
    for train_ex in train_dataset:
        train_example_strings.append(example_to_string(train_ex))

    sorted_indices_in_subset = []
    err_indices = []

    for idx in tqdm(subset_indices):
        sim_to_other_train_exs = train_to_train_sim[idx]
        sorted_indices = [i for i in range(len(sim_to_other_train_exs))]
        sorted_indices = sorted(sorted_indices, key = lambda x: -sim_to_other_train_exs[x])
        prompt_examples = [train_example_strings[sorted_indices[i]] for i in range(5)]
        prompt_examples.reverse() # So that the most similar example is last
        prompt = ""
        for prompt_example in prompt_examples:
            prompt += prompt_example + "\n\n"
        prompt += "Question: " + train_dataset[idx]["question"] + "\nAnswers:"
        max_new_tokens = 30 if data_name == "ambiqa" else 75 if data_name == "quest" else 100
        kwargs = {"prompts" : [prompt], "min_new_tokens" : 2, "max_new_tokens" : max_new_tokens, "num_beams" : 5, "eos_token_id" : 13}
        kwargs_str = json.dumps(kwargs)
        params = {"kwargs" : kwargs_str}
        response = requests.get("http://127.0.0.1:"+port+"/", params=params)
        if True:
            try:
                response = json.loads(response.text)
                output_text = response[0]["output_text"]  
                output_text = output_text.split("\n")[0].strip().lower()
            except: 
                output_text = ""
                err_indices.append(idx)
            pred_answers = output_text.split("|")
            pred_answers = [pred_answers[i].strip() for i in range(len(pred_answers))]
            pred_answers = list(set(pred_answers))
            gt_answers = train_dataset[idx]["answers"]
            gt_answers = [gt_answers[i].strip().lower() for i in range(len(gt_answers))]
            gt_answers = list(set(gt_answers))
            score = score_generation_output(pred_answers, gt_answers)
            sorted_indices_in_subset.append((idx, score))

    print(err_indices)
    sorted_indices_in_subset = sorted(sorted_indices_in_subset, key=lambda x: x[1]["EM"]["F1"])
    # Save this entire object to disk
    sorted_indices_in_subset_f = open("../jsons/"+data_name+"/sorted_indices_in_subset.json", "w")
    json.dump(sorted_indices_in_subset, sorted_indices_in_subset_f, indent=4)
    sorted_indices_in_subset_f.close()

def observe_scores(data_name, some_pk_thresh):
    sorted_indices_in_subset_f = open("../jsons/"+data_name+"/sorted_indices_in_subset.json")
    sorted_indices_in_subset = json.load(sorted_indices_in_subset_f)
    sorted_indices_in_subset_f.close()

    zero, half, one = 0, 0, 0
    half_idx = -1

    train_dataset_f = open("../jsons/"+data_name+"/train_dataset.json")
    train_dataset = json.load(train_dataset_f)
    train_dataset_f.close()
    n_group = 2 if data_name == "quest" else 4
    n_example = 5

    zeros, ones, halfs = [], [], []
    for i in range(len(sorted_indices_in_subset)):
        if (data_name == "qampari") and (("date" in train_dataset[sorted_indices_in_subset[i][0]]["question"]) or ("dates" in train_dataset[sorted_indices_in_subset[i][0]]["question"])): continue
        if sorted_indices_in_subset[i][1]["EM"]["F1"] == 0: zeros.append(i)
        elif sorted_indices_in_subset[i][1]["EM"]["F1"] == 1.0: ones.append(i)
        if sorted_indices_in_subset[i][1]["EM"]["F1"] >= 0.5:
            if len(halfs) < some_pk_thresh: halfs.append(i)
            if (len(halfs) == some_pk_thresh) and (half_idx == -1): half_idx = i

    return zeros, halfs, ones

def pick_out_three_prompt_sets(data_name):
    sorted_indices_in_subset_f = open("../jsons/"+data_name+"/sorted_indices_in_subset.json")
    sorted_indices_in_subset = json.load(sorted_indices_in_subset_f)
    sorted_indices_in_subset_f.close()

    train_dataset_f = open("../jsons/"+data_name+"/train_dataset.json")
    train_dataset = json.load(train_dataset_f)
    train_dataset_f.close()

    n_group, n_example = 2 if data_name == "quest" else 4, 5
    threshold = 20
    zeros, halfs, ones = observe_scores(data_name, n_group * n_example)

    sets = {}
    diff = threshold - n_group*n_example

    none_pk_idx = random.sample(zeros, k=threshold)
    for j in range(4):
        none_pk_set = []
        for i in range(n_example):
            none_pk_set.append(train_dataset[sorted_indices_in_subset[none_pk_idx[0+j*n_example+i]][0]])
        sets["none_pk_set"+str(j+1)] = none_pk_set

    some_pk_idx = random.sample(halfs, k=len(halfs))
    for j in range(n_group):
        some_pk_set = []
        for i in range(n_example):
            some_pk_set.append(train_dataset[sorted_indices_in_subset[some_pk_idx[j*n_example+i]][0]])
        sets["some_pk_set"+str(j+1)] = some_pk_set

    if len(ones) < threshold:
        ones = [i for i in range(len(sorted_indices_in_subset)-threshold, len(sorted_indices_in_subset))]
    full_pk_idx = random.sample(ones, k=threshold)
    for j in range(n_group):
        full_pk_set = []
        for i in range(n_example):
            full_pk_set.append(train_dataset[sorted_indices_in_subset[full_pk_idx[diff+j*n_example+i]][0]])
        sets["full_pk_set"+str(j+1)] = full_pk_set

    baseline_indices = random.sample([i for i in range(len(train_dataset))], k=threshold)
    for j in range(4):
        base_pk_set = []
        for i in range(n_example):
            base_pk_set.append(train_dataset[baseline_indices[0+j*n_example+i]])
        sets["base_pk_set"+str(j+1)] = base_pk_set

    prompt_sets_f = open("../"+data_name+"_experiments/answer_set/answer_sets.json", "w")
    json.dump(sets, prompt_sets_f, indent=4)
    prompt_sets_f.close()

    idxs = [sorted_indices_in_subset[i][0] for i in none_pk_idx]
    idxs.extend([sorted_indices_in_subset[i][0] for i in some_pk_idx])
    idxs.extend([sorted_indices_in_subset[i][0] for i in full_pk_idx])
    idxs.extend([i for i in baseline_indices])
    print([sorted_indices_in_subset[i][0] for i in none_pk_idx])
    print([sorted_indices_in_subset[i][0] for i in some_pk_idx])
    print([sorted_indices_in_subset[i][0] for i in full_pk_idx])
    print(baseline_indices)
    return idxs

if __name__ == "__main__":
    data_name = sys.argv[1] # ['ambiqa', 'qampari', 'quest']
    if data_name not in ['ambiqa', 'quest', 'qampari']: raise Exception('invalid data_name')
    port = sys.argv[2]

    plot_train_against_train(data_name)
    eval_on_train_subset(data_name, port)
    ids = pick_out_three_prompt_sets(data_name)