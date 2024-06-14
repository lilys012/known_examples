import json

filepaths = [
    "../dataset/qampari/train_data.jsonl",
    "../dataset/qampari/dev_data.jsonl",
    "../dataset/qampari/test_data.jsonl"
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
        formatted_entry["question"] = entry["question_text"]
        answer_list = entry["answer_list"]
        answer_list = sorted(list(set([answer_list[i]["answer_text"] for i in range(len(answer_list))])))
        for i in range(len(answer_list)):
            if len(answer_list[i]) >= 2 and answer_list[i][:2] == "[[" and answer_list[i][-2:] == "]]":
                answer_list[i] = answer_list[i][2:-2]
        formatted_entry["answers"] = answer_list
        formatted_data.append(formatted_entry)

    if dest_filepath == "train_dataset.json":
        new_train = []
        for example in formatted_data:
            if example not in new_train:
                new_train.append(example)
        formatted_data = new_train

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