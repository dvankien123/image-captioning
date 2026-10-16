"""Flickr30k caption dataset and variable-length batch collation."""

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from src.utils import load_splits

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, json_dir, vocab, transform, caption_dir=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.vocab = vocab
        self.transform = transform

        split = load_splits(self.json_dir)
        if not isinstance(split, dict):
            raise ValueError(f"Split file must contain a filename-to-captions object: {self.json_dir}")

        self.samples = []
        for image_name, captions in split.items():
            if isinstance(captions, str):
                captions = [captions]
            if not isinstance(captions, list):
                raise ValueError(f"Captions for {image_name!r} must be a string or list")
            self.samples.extend(
                (image_name, caption.strip())
                for caption in captions
                if isinstance(caption, str) and caption.strip()
            )

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

