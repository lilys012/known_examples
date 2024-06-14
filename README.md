# Known vs Unknown Examples

This is the repository for the paper [Crafting In-context Examples according to LMs’ Parametric Knowledge](https://lilys012.github.io/assets/pdf/craftingIE.pdf) (NAACL 2024 Findings).

## [/dataset](/dataset)

We use 4 datasets, [AmbigQA](https://nlp.cs.washington.edu/ambigqa/), [QAMPARI](https://samsam3232.github.io/qampari/), [QUEST](https://github.com/google-research/language/tree/master/language/quest), and [GSM8K](https://github.com/openai/grade-school-math). Please place each dataset under [dataset](/dataset) directory with designated namings.

## [/jsons](/jsons)

This directory contains preprocessed datasets and experimental results. To preprocess three multi-QA datasets, run `process.py` in the folder of each dataset name. This will create `train.json`, `dev.json`, and `test.json` (if exists). Furthermore, `train_reduced.json` will be created, which contains examples that have less than 20 answers.

## [score_with_simcse.py](score_with_simcse.py)

As our experiments utilize the SimCSE embeddings, we compute the embeddings for the datasets in prior. For each dataset, run the command below.

```
python score_with_simcse.py [data_name]
```

This will result `train_embeddings.pt`, `dev_embeddings.pt`, and `test_embeddings.pt` (if exists), and `train_reduced_embeddings.pt`.

## [/web_server](/web_server)

This is a critical folder than contains code for serving the language model for inference. Rather than loading LLM to the GPUs every time we run an experiment, we load them once by running the web server, and query them in our experiment scripts. The web server allows for complex types of queries (i.e. much more than basic text completion), many of which are useful for several of the experiments. We support two LLMs, Llama2 and OPT. You can run the web server by the command below.

```
flask --app app_run_[model] run -p [PORT]
```

## [/answer_set](/answer_set)

This directory conducts experiment in Section 3. We first construct in-context example sets, differed by the amount of parametric knowledge the model shows. Please run [answer_set.py](/answer_set/answer_set.py) with the command below.

```
python answer_shot.py [data_name] [strategy] [split] [PORT]
```

Once we have obtained `answer_sets.json`, we now use these to infer on evaluation datasets with [answer_shot.py](/answer_set/answer_shot.py). `strategy` includes `none, some, full, base`, which responds to `Unknown, HalfKnown, Known, Random`, respectively.

```
python answer_set.py [data_name] [PORT]
```

## [/answer_set/math](/answer_set/math)

We also experiment on GSM8K dataset. [obtain_f1.py](/answer_set/math/obtain_f1.py) and [calc_f1.py](/answer_set/math/calc_f1.py) evaluate the training examples with [intial_prompts](/answer_set/math/initial_prompts.json) and construct in-context example sets. Now we infer on test data with [answer_shot.py](/answer_set/math/answer_shot.py) and compute accuracy with [eval.py](/answer_set/math/eval.py). For `INST`, please use `none, some, full, base`. `Set_num` ranges from 1 to 4.

```
python obtain_f1.py [PORT]
python calc_f1.py
python answer_shot.py [PORT] [INST] [Set_num]
python eval.py [INST] [Set_num]
```

## [/answer_ordering](/answer_ordering)

We first order the gold answers of train examples with diverse ordering strategies. For `Greedy Ordering`, please run the command below. `reverse`, which takes a value among `grd` and `rev`, indicates whether its ordering is `Greedy` or `Reverse Greedy`.

```
python greedy_decoding.py [data_name] [PORT] [reverse]
```

For `Perplexity Ordering`, please run the command below. Unlike above, we can obtain both `Perplexity` and `Reverse Perplexity` orderings with one inference. They are denoted as `asc` and `desc`, respectively.

```
python perplexity.py [data_name] [PORT]
```

Now that we have the answer orderings of train examples ready, we can infer on the evaluation datasets. `order` accepts a value among `rand, grd, asc, rev, desc, alpha`, which represents six answer ordering stratgies introduced in our paper. They are in the order as presented in Table 4.

```
python answer_ordering.py [data_name] [PORT] [split] [order]
```

## [experiment.py](experiment.py)

Finally, we evaluate the generated outputs. [experiment.py](experiment.py) parses the model generations and computes F1/EM scores. The first argument of this script indicates which experiment it is, since we have two independent ones. `pk` indicates experiments from [answer_set](/answer_set) and `ord` indicates experiments from [answer_ordering](/answer_ordering).

```
python experiment.py [pk] [data_name] [split] [strategy] [Set_num]
python experiment.py [ord] [data_name] [split] [order]
```
