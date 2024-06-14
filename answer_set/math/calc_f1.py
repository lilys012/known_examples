import json
from dataset import get_examples, extract_answer
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
import requests
from tqdm import tqdm 
import random

exp_log = []
invalids = []
wrongs = []
corrects = []
f = open(f"../../jsons/math_experiments/initial_gen.json")
data = json.load(f)
for ex in data:
    ex_log = ex
    pred_ans = extract_answer(ex["output_text"])
    real_ans = extract_answer(ex["answer"])

    if pred_ans == "[invalid]": 
        ex_log["score"] = -1
        invalids.append(ex["example_idx"])
    elif pred_ans == real_ans: 
        ex_log["score"] = 1
        corrects.append(ex["example_idx"])
    else: 
        ex_log["score"] = 0
        wrongs.append(ex["example_idx"])
    exp_log.append(ex_log)

print(len(corrects), len(wrongs), len(invalids))

n_prompts, n_sets = 6, 4
unknown_idxs = random.sample(wrongs, n_prompts*n_sets+int(n_prompts/2)*n_sets+n_prompts*n_sets)
known_idxs = random.sample(corrects, n_prompts*n_sets+int(n_prompts/2)*n_sets)
known_sets, unknown_sets, halfknown_sets, rand_sets = {}, {}, {}, {}
for i in range(n_sets):
    prompt = ""
    idxs = known_idxs[i*n_prompts:(i+1)*n_prompts]
    for j in idxs: prompt += "Question: " + exp_log[j]["question"] + "\nAnswer: " + exp_log[j]["answer"].replace('\n', ' ') + "\n\n"
    known_sets[f"full_{i+1}"] = {
        "indices" : idxs,
        "prompt" : prompt
    }

    prompt = ""
    idxs = unknown_idxs[i*n_prompts:(i+1)*n_prompts]
    for j in idxs: prompt += "Question: " + exp_log[j]["question"] + "\nAnswer: " + exp_log[j]["answer"].replace('\n', ' ') + "\n\n"
    unknown_sets[f"none_{i+1}"] = {
        "indices" : idxs,
        "prompt" : prompt
    }

    prompt = ""
    idxs = unknown_idxs[n_prompts*n_sets+int(n_prompts/2)*i:n_prompts*n_sets+int(n_prompts/2)*(i+1)]
    idxs.extend(known_idxs[n_prompts*n_sets+int(n_prompts/2)*i:n_prompts*n_sets+int(n_prompts/2)*(i+1)])
    idxs = random.sample(idxs, len(idxs))
    for j in idxs: prompt += "Question: " + exp_log[j]["question"] + "\nAnswer: " + exp_log[j]["answer"].replace('\n', ' ') + "\n\n"
    halfknown_sets[f"some_{i+1}"] = {
        "indices" : idxs,
        "prompt" : prompt
    }

random_idxs = random.sample(range(len(exp_log)), n_prompts*n_sets)
for i in range(n_sets):
    prompt = ""
    idxs = random_idxs[i*n_prompts:(i+1)*n_prompts]
    for j in idxs: prompt += "Question: " + exp_log[j]["question"] + "\nAnswer: " + exp_log[j]["answer"].replace('\n', ' ').rstrip() + "\n\n"
    rand_sets[f"base_{i+1}"] = {
        "indices" : idxs,
        "prompt" : prompt
    }


f = open(f"../../jsons/math_experiments/gen_score.json", "w")
json.dump(exp_log, f, indent=4)
f.close()

f = open(f"../../jsons/math_experiments/prompts.json", "w")
json.dump([unknown_sets, halfknown_sets, known_sets, rand_sets], f, indent=4)
f.close()