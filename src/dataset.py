import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, caption_dir,
                 json_dir, vocab, transform):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.json_dir = json_dir
        self.vocab = vocab
        self.transform = transform

        df = pd.read_csv(self.caption_dir, sep = '|')
        df.columns = df.columns.str.strip()

        with open(self.json_dir, "r") as f:
            data = json.load(f)

        df = df[df["image_name"].isin(data)]
        df = df.dropna(subset=['comment'])
        self.samples = df[["image_name", "comment"]].to_numpy().tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption_ids = self.vocab.numericalize(caption)
        caption_tensor = torch.tensor(caption_ids, dtype = torch.long)

        return image, caption_tensor

def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images, dim = 0)

    lengths = torch.tensor([len(cap) for cap in captions])
    captions_padded = pad_sequence(captions, batch_first = True, padding_value = 0)

    return images, captions_padded, lengths

