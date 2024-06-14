import os
import json
from tqdm import tqdm
import torch
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

experiment = sys.argv[1]
if experiment not in ['pk', 'ord']: raise Exception('invalid experiment')
data_name = sys.argv[2]
if data_name not in ['ambiqa', 'quest', 'qampari']: raise Exception('invalid data_name')
split = sys.argv[3]
if (data_name == 'ambiqa' and split != 'dev') or split not in ['dev', 'test']: raise Exception('invalid split')

if experiment == 'pk':
    strategy = sys.argv[4]
    if strategy not in ['none', 'some', 'full', 'base']: raise Exception('invalid strategy')
    set_num = int(sys.argv[5])
    user_filepath = "jsons/"+data_name+"_experiments/answer_set/"+strategy+"/"+split+"_answer_set"+"_"+str(set_num)+".json"
elif experiment == 'ord':
    order = sys.argv[4] # ['rev', 'grd', 'rand', 'alpha', 'asc', 'desc']
    save_path = "jsons/"+data_name+"_experiments/answer_ordering/"+split+"_"+order+".json"

def word_level_f1(str1, str2):
    str1_words = str1.split()
    str2_words = str2.split()
    if len(str1_words) == 0 or len(str2_words) == 0:
        return 0
    word_recall = 0
    for str1_word in str1_words:
        for str2_word in str2_words:
            if str1_word.lower() == str2_word.lower():
                word_recall += 1
                break
    word_recall /= len(str1_words)
    word_precision = 0
    for str2_word in str2_words:
        for str1_word in str1_words:
            if str1_word.lower() == str2_word.lower():
                word_precision += 1
                break
    word_precision /= len(str2_words)
    if (word_recall + word_precision) != 0:
        word_f1 = 2 * word_recall * word_precision / (word_recall + word_precision)
    else:
        word_f1 = 0
    return word_f1

def score_generation_output(pred_answers, gt_answers):
    score = {
        "EM" : {
            "F1" : 0, # Set level F1
            "P" : 0, # Set level precision
            "R" : 0 # Set level recall
        },
        "F1" : {
            "F1" : 0,
            "P" : 0,
            "R" : 0
        }
    }

    assert not (len(pred_answers) == 0 or len(gt_answers) == 0)

    for pred_answer in pred_answers:
        max_word_level_f1 = 0
        for gt_answer in gt_answers:
            if pred_answer == gt_answer:
                score["EM"]["P"] += 1
            word_f1 = word_level_f1(pred_answer, gt_answer)
            if word_f1 > max_word_level_f1:
                max_word_level_f1 = word_f1
        score["F1"]["P"] += max_word_level_f1
    score["EM"]["P"] /= len(pred_answers)
    score["F1"]["P"] /= len(pred_answers)

    for gt_answer in gt_answers:
        max_word_level_f1 = 0
        for pred_answer in pred_answers:
            if pred_answer == gt_answer:
                score["EM"]["R"] += 1
            word_f1 = word_level_f1(pred_answer, gt_answer)
            if word_f1 > max_word_level_f1:
                max_word_level_f1 = word_f1
        score["F1"]["R"] += max_word_level_f1
    score["EM"]["R"] /= len(gt_answers)
    score["F1"]["R"] /= len(gt_answers)

    if score["EM"]["P"] + score["EM"]["R"] != 0:
        score["EM"]["F1"] = 2 * score["EM"]["P"] * score["EM"]["R"] / (score["EM"]["P"] + score["EM"]["R"])
    if score["F1"]["P"] + score["F1"]["R"] != 0:
        score["F1"]["F1"] = 2 * score["F1"]["P"] * score["F1"]["R"] / (score["F1"]["P"] + score["F1"]["R"])
    return score

def score_experiment(filepath, split="dev"):
    log_f = open(filepath)
    log = json.load(log_f)
    log_f.close()

    experiment_score = {
        "EM" : {
            "F1" : 0,
            "P" : 0,
            "R" : 0
        },
        "F1" : {
            "F1" : 0,
            "P" : 0,
            "R" : 0
        },
    }

    complete_exact_match_counts = 0
    n_pred_answer_full, n_pred_answer_unq, n_gt_answer_full, n_gt_answer_unq = 0, 0, 0, 0
    for i, elem in enumerate(log):
        generation_output = elem["output_text"]
        generation_output = generation_output.split("\n")[0].strip().lower()
        generation_output = generation_output.lstrip("|").rstrip("|").strip()
        pred_answers = generation_output.split("|")
        pred_answers = [pred_answers[i].strip() for i in range(len(pred_answers))]
        pred_first_answer, pred_last_answer = pred_answers[0], pred_answers[-1]
        n_pred_answer_full += len(pred_answers)
        pred_answers = list(set(pred_answers))
        n_pred_answer_unq += len(pred_answers)

        gt_answers = elem[split + "_example"]["answers"]
        gt_answers = [gt_answers[i].strip().lower() for i in range(len(gt_answers))]
        n_gt_answer_full += len(gt_answers)
        gt_answers = list(set(gt_answers))
        n_gt_answer_unq += len(gt_answers)

        score = score_generation_output(pred_answers, gt_answers)

        experiment_score["EM"]["F1"] += score["EM"]["F1"]
        experiment_score["EM"]["P"] += score["EM"]["P"]
        experiment_score["EM"]["R"] += score["EM"]["R"]
        experiment_score["F1"]["F1"] += score["F1"]["F1"]
        experiment_score["F1"]["P"] += score["F1"]["P"]
        experiment_score["F1"]["R"] += score["F1"]["R"]
        if score["EM"]["F1"] == 1: complete_exact_match_counts += 1

    loglen = len(log)
    experiment_score["EM"]["F1"] /= loglen
    experiment_score["EM"]["P"] /= loglen
    experiment_score["EM"]["R"] /= loglen
    experiment_score["F1"]["F1"] /= loglen
    experiment_score["F1"]["P"] /= loglen
    experiment_score["F1"]["R"] /= loglen

    experiment_score["Hard EM"] = complete_exact_match_counts / loglen
    experiment_score["First EM"] = first_answer_exact_match_count / loglen
    experiment_score["Last EM"] = last_answer_exact_match_count / loglen
    experiment_score["n_ans"] = {
        "pred_full" : n_pred_answer_full / loglen,
        "pred_unq" : n_pred_answer_unq / loglen,
        "gt_full" : n_gt_answer_full / loglen,
        "gt_unq" : n_gt_answer_unq / loglen
    }

    return experiment_score

sc = score_experiment(user_filepath, split=split)
print("EM: "+str(round(sc["Hard EM"], 3))+" EM-P: "+str(round(sc["EM"]["P"], 3))+" EM-R: "+str(round(sc["EM"]["R"], 3))+" EM-F1: "+str(round(sc["EM"]["F1"], 3))+" First EM: "+str(round(sc["First EM"], 3))+" Last EM: "+str(round(sc["Last EM"], 3)))
print("F1-F1: "+str(round(sc["F1"]["F1"], 3))+" Pred Unq: "+str(round(sc["n_ans"]["pred_unq"], 3)))