import json

filepaths = [
    "../dataset/quest/train.jsonl",
    "../dataset/quest/val.jsonl",
    "../dataset/quest/test.jsonl"
]

dest_filepaths = [
    "train_dataset.json",
    "dev_dataset.json",
    "test_dataset.json"
]

for filepath, dest_filepath in zip(filepaths, dest_filepaths):
    f = open(filepath, "r")
    data = list(f)
    f.close()

    for i in range(len(data)):
        data[i] = json.loads(data[i])

    formatted_data = []
    for entry in data:
        formatted_entry = {}
        formatted_entry["question"] = entry["query"]
        answer_set = entry["docs"]
        for i in range(len(answer_set)):
            answer = answer_set[i]
            if "(" in answer:
                answer_set[i] = answer[:answer.index("(")-1] # Remove Wikipedia disambiguations
        formatted_entry["answers"] = sorted(list(set(answer_set))) # Alphabetize

        if len(formatted_entry["answers"]) <= 1:
            continue # We only evaluate on multi-answer questions

        formatted_data.append(formatted_entry)

    f = open(dest_filepath, "w")
    json.dump(formatted_data, f, indent=4)
    f.close()

f1 = open("train_dataset.json", "r")
train = json.load(f1)

new_train = []
for example in train:
    if example not in new_train:
        if len(example["answers"]) <= 20: new_train.append(example)

f2 = open("train_dataset_reduced.json", "w")
json.dump(new_train, f2, indent=4)

print(len(train), len(new_train))