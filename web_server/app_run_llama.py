import os
from transformers import LlamaTokenizer, LlamaForCausalLM
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from flask import Flask
from flask import request
import copy
from math import log2

model_path = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token='[PAD]'
model = LlamaForCausalLM.from_pretrained(model_path)

model = model.half()

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

    # force generation
    if "force_generation" in kwargs:
        text_to_generate = kwargs["force_generation"]
        del kwargs["force_generation"]
        tokenized_text_to_generate = tokenizer(text_to_generate, return_tensors="pt").input_ids[0][1:]
        kwargs["max_new_tokens"] = 1
        kwargs["min_new_tokens"] = 1
        log_probs = [[] for _ in range(len(prompts))]
        new_prompts = copy.deepcopy(prompts)
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
        sequences = [[] for _ in range(len(prompts))]
        for i, tokenized_prompt in enumerate(tokenized_prompts):
            for tkn in tokenized_prompt:
                sequences[i].append(tkn)
        for i in range(len(tokenized_text_to_generate)):
            model_input = tokenizer(new_prompts, return_tensors="pt", padding=True).to(device)
            model_output = model.generate(**model_input, **kwargs, output_scores=True, return_dict_in_generate=True)
            assert len(model_output.scores) == 1
            scores = model_output.scores[0]
            for prompt_idx in range(len(scores)):
                logits = scores[prompt_idx]
                lps = nn.functional.log_softmax(logits)
                log_probs[prompt_idx].append(lps[tokenized_text_to_generate[i]].item())
            decoded_token = tokenizer.decode(tokenized_text_to_generate[i])
            new_prompts = [new_prompts[j] + decoded_token for j in range(len(new_prompts))]
            for j in range(len(sequences)):
                sequences[j].append(tokenized_text_to_generate[i])
        for i in range(len(sequences)):
            sequences[i] = torch.tensor(sequences[i])
        kwargs["force_generation"] = text_to_generate # add it back
    elif "answer_ordering" in kwargs:
        assert len(prompts) == 1
        prompt = prompts[0]
        answer_ordering_params = kwargs["answer_ordering"]
        del kwargs["answer_ordering"]
        if answer_ordering_params["type"] == "greedy_decoding" or answer_ordering_params["type"] == "reverse_greedy_decoding":
            answer_set = answer_ordering_params["answer_set"]
            # tokenize each of the answers in the answer set
            tokenized_answer_set = []
            for answer in answer_set:
                tokenized_answer_set.append(tokenizer(answer, return_tensors="pt").input_ids[0][1:])
            kwargs["max_new_tokens"] = 1
            kwargs["min_new_tokens"] = 1
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
                    model_output = model.generate(**model_input, **kwargs, output_scores=True, return_dict_in_generate=True)
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
            kwargs["max_new_tokens"] = 1
            kwargs["min_new_tokens"] = 1
            perplexity = [] # proportional to sigma(logP) / -N
            for answer in answer_set:
                running_prompt = prompt
                running_prompt_tokenized = tokenizer(running_prompt, return_tensors="pt").input_ids[0][1:]
                running_prompt_tokenized = [running_prompt_tokenized[i].item() for i in range(len(running_prompt_tokenized))]
                tokenized_answer_set = tokenizer(answer, return_tensors="pt").input_ids[0][1:]
                perp = 0
                for token in tokenized_answer_set:
                    model_input = tokenizer(running_prompt, return_tensors="pt").to(device)
                    model_output = model.generate(**model_input, **kwargs, output_scores=True, return_dict_in_generate=True)
                    assert len(model_output.scores) == 1
                    perp += log2(torch.nn.functional.softmax(model_output.scores[0][0])[token].item())
                    running_prompt_tokenized.append(token)
                    running_prompt = tokenizer.decode(running_prompt_tokenized)
                perplexity.append(-perp / len(tokenized_answer_set))

            answer_from_largest = [[y, x] for y, x in sorted(zip(perplexity, answer_set), key=lambda p: -p[0])]
            return {"descend": answer_from_largest}
        else: pass 
    else:
        model_input = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        model_output = model.generate(**model_input, **kwargs, return_dict_in_generate=True)
        sequences = model_output.sequences

    to_return = []
    for i, sequence in enumerate(sequences):
        indices = []
        for j, elem in enumerate(sequence):
            if elem.item() > 0:
                indices.append(j)
        sequence_parsed = sequence[indices]
        text = tokenizer.decode(sequence_parsed)
        d = {"text" : text}
        d["sequence"] = sequence_parsed.cpu().numpy().tolist()
        if "force_generation" in kwargs:
            scores = log_probs[i]
            d["scores"] = scores
        else:
            output_sequence = sequence[len(model_input.input_ids[0]):]
            output_indices = []
            for j, elem in enumerate(output_sequence):
                if elem.item() > 0:
                    output_indices.append(j)
            output_sequence_parsed = output_sequence[output_indices]
            output_text = tokenizer.decode(output_sequence_parsed)
            d["output_text"] = output_text
        to_return.append(d)

    return to_return