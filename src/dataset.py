"""Dataset loaders and variable-length batch collation for Flickr30k and MS COCO."""

import json
from pathlib import Path
import torch

from src.utils import load_splits
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class Flickr30kDataset(Dataset):
    def __init__(self, image_path, split_path, vocab, transform):
        self.image_path = Path(image_path)
        self.split_path = Path(split_path)
        self.vocab = vocab
        self.transform = transform

        data = load_splits(self.split_path)

        # Each image may have several captions; retain every caption for split images.
        self.samples = []
        for image_name, captions in data.items():
            for caption in captions:
                self.samples.append((image_name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        image_path = self.image_path / image_name
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption_ids = self.vocab.numericalize(caption)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor


class CocoDataset(Dataset):
    """MS COCO 2017 caption dataset."""

    def __init__(self, image_path, annotation_path, vocab, transform):
        self.image_path = Path(image_path)
        self.annotation_path = Path(annotation_path)
        self.vocab = vocab
        self.transform = transform

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

        self.samples = []
        for ann in coco_data["annotations"]:
            image_name = id_to_filename.get(ann["image_id"], f"{ann['image_id']:012d}.jpg")
            self.samples.append((image_name, ann["caption"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        image_path = self.image_path / image_name
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption_ids = self.vocab.numericalize(caption)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)

    lengths = torch.tensor([len(cap) for cap in captions])
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions_padded, lengths
