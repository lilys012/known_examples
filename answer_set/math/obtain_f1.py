import json
from dataset import get_examples
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
import requests
from tqdm import tqdm 
import os

port = sys.argv[1]

initial_prompts_f = open("initial_prompts.json")
initial_prompts = json.load(initial_prompts_f)
initial_prompts_f.close()
cot_prompt = "\n\n".join(initial_prompts)

train_dataset = get_examples("train")

errs = []
experiment_log = []
for i in tqdm(range(len(train_dataset))):
    example_log = {}
    example_log["example_idx"] = i
    example_log["question"] = train_dataset[i]["question"].rstrip()
    example_log["answer"] = train_dataset[i]["answer"]

    prompt = cot_prompt + "\n\nQuestion: " + train_dataset[i]["question"].rstrip() + "\nAnswer:"
    example_log["prompt"] = prompt
    kwargs = {"prompts" : [prompt], "min_new_tokens" : 2, "max_new_tokens" : 150, "num_beams" : 5, "eos_token_id" : 13}
    kwargs_str = json.dumps(kwargs)
    params = {"kwargs" : kwargs_str}
    response = requests.get("http://127.0.0.1:"+port+"/", params=params)

    try:
        response = json.loads(response.text)
        output_text = response[0]["output_text"]
    except:
        output_text = ""
        errs.append(i)

    example_log["output_text"] = output_text
    experiment_log.append(example_log)

os.makedirs("../../jsons/math_experiments", exist_ok=True)
f = open(f"../../jsons/math_experiments/initial_gen.json", "w")
json.dump(experiment_log, f, indent=4)
f.close()

print(errs)