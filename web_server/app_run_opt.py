import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from flask import Flask
from flask import request
import copy
from math import log, log2
import gc

model_path = "facebook/opt-13b"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token='[PAD]'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
device = torch.device("cuda:0")
model.to(device)

app = Flask(__name__)

@app.route("/")
def query_model():
    global tokenizer, model, device
    kwargs_str = request.args.get("kwargs", type=str)
    kwargs = json.loads(kwargs_str)

    # prompts
    prompts = kwargs["prompts"]
    del kwargs["prompts"]
    prompt = prompts[0]

    if "answer_ordering" in kwargs:
        assert len(prompts) == 1
        answer_ordering_params = kwargs["answer_ordering"]
        if answer_ordering_params["type"] == "greedy_decoding" or answer_ordering_params["type"] == "reverse_greedy_decoding":
            answer_set = answer_ordering_params["answer_set"]
            # tokenize each of the answers in the answer set
            tokenized_answer_set = []
            for answer in answer_set:
                tokenized_answer_set.append(tokenizer(answer, return_tensors="pt").input_ids[0][1:])
            gen_kwargs = {"max_new_tokens":1, "min_new_tokens":1}
            running_prompt = prompt
            running_prompt_tokenized = tokenizer(running_prompt, return_tensors="pt").input_ids[0][1:]
            running_prompt_tokenized = [running_prompt_tokenized[i].item() for i in range(len(running_prompt_tokenized))]
            cnt = 0
            # pop each selected answer
            while len(tokenized_answer_set) > 0:
                remaining_answer_choices = copy.deepcopy(tokenized_answer_set)
                remaining_answer_choices_indices = [i for i in range(len(remaining_answer_choices))]
                while len(remaining_answer_choices) > 1:
                    model_input = tokenizer(running_prompt, return_tensors="pt").to(device)
                    model_output = model.generate(**model_input, **gen_kwargs, output_scores=True, return_dict_in_generate=True)
                    assert len(model_output.scores) == 1
                    scores = model_output.scores[0][0]
                    # Candidate tokens (first token of each remainders)
                    tokens_to_choose_from = list(set([remaining_answer_choices[i][0].item() for i in range(len(remaining_answer_choices))]))
                    scored_tokens_to_choose_from = [(tokens_to_choose_from[i], scores[tokens_to_choose_from[i]].item()) for i in range(len(tokens_to_choose_from))]
                    if answer_ordering_params["type"] == "greedy_decoding":
                        token_scoring_key = lambda x: -x[1]
                    else:
                        token_scoring_key = lambda x: x[1]
                    scored_tokens_to_choose_from = sorted(scored_tokens_to_choose_from, key=token_scoring_key)
                    # Select this token
                    chosen_token = scored_tokens_to_choose_from[0][0]
                    new_remaining_answer_choices = []
                    new_remaining_answer_choices_indices = []
                    answer_no_more_tokens, answer_no_more_tokens_idx = False, -1
                    # Find this token from remaining choices
                    for i, remaining_answer_choice in enumerate(remaining_answer_choices):
                        if remaining_answer_choice[0].item() == chosen_token:
                            # Remove this token from remaining candidates
                            remaining_answer_choice = remaining_answer_choice[1:]
                            # Select this answer
                            if len(remaining_answer_choice) == 0:
                                answer_no_more_tokens = True
                                answer_no_more_tokens_idx = remaining_answer_choices_indices[i]
                                break
                            # Prune this set
                            new_remaining_answer_choices.append(remaining_answer_choice)
                            new_remaining_answer_choices_indices.append(remaining_answer_choices_indices[i])
                    if cnt == 0: cnt = len(new_remaining_answer_choices)
                    running_prompt_tokenized.append(chosen_token)
                    running_prompt = tokenizer.decode(running_prompt_tokenized)
                    remaining_answer_choices = new_remaining_answer_choices
                    remaining_answer_choices_indices = new_remaining_answer_choices_indices
                    if answer_no_more_tokens:
                        # We will end the answer generation
                        remaining_answer_choices = []
                        remaining_answer_choices_indices = [answer_no_more_tokens_idx]
                        break
                if len(remaining_answer_choices) != 0: # the only value it can be is 1
                    for token in remaining_answer_choices[0]:
                        running_prompt_tokenized.append(token)
                    running_prompt = tokenizer.decode(running_prompt_tokenized)
                # Remove the answer that we've just added to the running prompt from tokenized_answer_set
                tokenized_answer_set = tokenized_answer_set[:remaining_answer_choices_indices[0]] + tokenized_answer_set[remaining_answer_choices_indices[0]+1:]
                if len(tokenized_answer_set) >= 1:
                    running_prompt += " |"
                    running_prompt_tokenized = tokenizer(running_prompt, return_tensors="pt").input_ids[0][1:]
                    running_prompt_tokenized = [running_prompt_tokenized[i].item() for i in range(len(running_prompt_tokenized))]
            # that's it, we have the answer: running prompt
            return {"text": running_prompt[len(prompt):].strip(), "cnt":cnt}
        elif answer_ordering_params["type"] == "perplexity":
            answer_set = answer_ordering_params["answer_set"]
            gen_kwargs = {"max_new_tokens":1, "min_new_tokens":1}
            perplexity = [] # proportional to sigma(logP) / -N
            for answer in answer_set:
                if True:
                    running_prompt = prompt
                    running_prompt_tokenized = tokenizer(running_prompt, return_tensors="pt").input_ids[0][1:]
                    running_prompt_tokenized = [running_prompt_tokenized[i].item() for i in range(len(running_prompt_tokenized))]
                    tokenized_answer_set = tokenizer(answer, return_tensors="pt").input_ids[0][1:]
                    perp = 0
                    for token in tokenized_answer_set:
                        model_input = tokenizer(running_prompt, return_tensors="pt").to(device)
                        model_output = model.generate(**model_input, **gen_kwargs, output_scores=True, return_dict_in_generate=True)
                        assert len(model_output.scores) == 1
                        try:
                            score = log(torch.nn.functional.softmax(model_output.scores[0][0], dim=-1)[token].item())
                        except:
                            norm_scores = model_output.scores[0][0] - max(model_output.scores[0][0])
                            score = float(norm_scores[token] - log(torch.sum(torch.exp(norm_scores))))
                        perp += score
                        running_prompt_tokenized.append(token)
                        running_prompt = tokenizer.decode(running_prompt_tokenized)
                    perplexity.append(-perp / len(tokenized_answer_set))

            answer_from_largest = [[y, x] for y, x in sorted(zip(perplexity, answer_set), key=lambda p: -p[0])]
            return {"descend": answer_from_largest}

    model_input = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    model_output = model.generate(**model_input, **kwargs, return_dict_in_generate=True)
    sequences = model_output.sequences

    to_return = []
    for i, sequence in enumerate(sequences):
        output_sequence = sequence[len(model_input.input_ids[0]):]
        output_indices = []
        for j, elem in enumerate(output_sequence):
            if elem.item() > 0:
                output_indices.append(j)
        output_sequence_parsed = output_sequence[output_indices]
        output_text = tokenizer.decode(output_sequence_parsed, skip_special_tokens=True)
        d = {"output_text": output_text}
        to_return.append(d)

    return to_return
