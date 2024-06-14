import json

train_path, train_dest_path = "../dataset/ambiqa/train_light.json", "train_dataset.json"
dev_path, dev_dest_path = "../dataset/ambiqa/dev_light.json", "dev_dataset.json"
paths = [[train_path, train_dest_path], [dev_path, dev_dest_path]]

for src_path, dest_path in paths:
    f = open(src_path)
    dataset = json.load(f)
    f.close()
    prompt_examples = []
    for example in dataset:
        question = example["question"]
        annotations = example["annotations"]
        answer_set = []
        if len(annotations) == 0:
            continue
        annotation = annotations[0] # For simplicity we will use only the first annotation
        if annotation["type"] == "singleAnswer":
            answer = annotation["answer"][0]
            answer_set.append(answer)
        else:
            qa_set = annotation["qaPairs"]
            for pair in qa_set:
                answer = pair["answer"][0]
                answer_set.append(answer)
        answer_set = sorted(list(set(answer_set))) # Alphabetize answers
        if len(answer_set) <= 1:
            continue # Our dataset will only consist of multi-answer questions
        example = {"question" : question, "answers" : answer_set}
        prompt_examples.append(example)

    f = open(dest_path, "w")
    json.dump(prompt_examples, f, indent = 4)
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