import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        packed = pack_padded_sequence(
            inputs, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )

        hiddens, _ = self.lstm(packed)
        outputs = self.fc(self.dropout(hiddens.data))
        return outputs
    def sample(self, features, max_len=30):
        batch_size = features.size(0)
        sampled_ids = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)

        inputs = features.unsqueeze(1)
        states = None

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            predicted = outputs.argmax(dim=1)
            for i in range(batch_size):
                if not finished[i]:
                    sampled_ids[i].append(predicted[i].item())
                    if predicted[i].item() == 2:  # end_idx
                        finished[i] = True
            if finished.all():
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids