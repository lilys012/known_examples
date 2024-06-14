import json
from dataset import get_examples, extract_answer
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
import requests
from tqdm import tqdm 
import random

inst = sys.argv[1]
set_num = int(sys.argv[2])

f = open(f"../../jsons/math_experiments/{inst}/gen_test_{inst}_{set_num}.json")
dataset = json.load(f)
f.close()

invalids = []
wrongs = []
corrects = []
for i, ex in enumerate(dataset):
    pred_ans = extract_answer(ex["output_text"])
    real_ans = extract_answer(ex["answer"])

    if pred_ans == "[invalid]": invalids.append(ex["example_idx"])
    elif pred_ans == real_ans: wrongs.append(ex["example_idx"])
    else: corrects.append(ex["example_idx"])

print(len(corrects), len(wrongs), len(invalids))
