import json

def karpathy_split():
    with open('../data/raw/dataset_flickr30k.json', 'r') as f:
        data = json.load(f)
    train = []
    val = []
    test = []
    for item in data["images"]:
        split = item["split"]
        if split == "train":
            train.append(item["filename"])
        elif split == "val":
            val.append(item["filename"])
        elif split == "test":
            test.append(item["filename"])
    return train, val, test

def save(train, val, test):
    with open('../data/splits/train.json', 'w') as f:
        json.dump(train, f)
    with open('../data/splits/val.json', 'w') as f:
        json.dump(val, f)
    with open('../data/splits/test.json', 'w') as f:
        json.dump(test, f)

@staticmethod
def load():
    with open('../data/splits/train.json', 'r') as f:
        train = json.load(f)
    with open('../data/splits/val.json', 'r') as f:
        val = json.load(f)
    with open('../data/splits/test.json', 'r') as f:
        test = json.load(f)
    return train, val, test