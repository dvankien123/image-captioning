"""Flickr30k caption dataset and variable-length batch collation."""

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
        caption_tensor = torch.tensor(caption_ids, dtype = torch.long)

        return image, caption_tensor

def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images, dim = 0)

    lengths = torch.tensor([len(cap) for cap in captions])
    captions_padded = pad_sequence(captions, batch_first = True, padding_value = 0)

    return images, captions_padded, lengths
