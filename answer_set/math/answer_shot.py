import json
from dataset import get_examples
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
import requests
from tqdm import tqdm 

port = sys.argv[1]
inst = sys.argv[2]
set_num = int(sys.argv[3])

inst_dict = {"none":0, "some":1, "full":2, "base":3}

prompts_f = open("../../jsons/math_experiments/prompts.json")
prompts = json.load(prompts_f)
prompts_f.close()
base_prompt = prompts[inst_dict[inst]][inst+"_"+str(set_num)]["prompt"]

test_dataset = get_examples("test")

errs = []
experiment_log = []
for i in tqdm(range(len(test_dataset))):
    example_log = {}
    example_log["example_idx"] = i
    example_log["question"] = test_dataset[i]["question"].rstrip()
    example_log["answer"] = test_dataset[i]["answer"]

    prompt = base_prompt + "Question: " + test_dataset[i]["question"].rstrip() + "\nAnswer:"
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

f = open(f"../../jsons/math_experiments/{inst}/gen_test_{inst}_{set_num}.json", "w")
json.dump(experiment_log, f, indent=4)
f.close()

print(errs)
