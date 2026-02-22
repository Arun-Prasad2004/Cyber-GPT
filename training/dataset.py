import torch
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer

class CyberDataset(IterableDataset):
    def __init__(self, file_path, tokenizer_path, block_size=512):
        self.file_path = file_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.block_size = block_size

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer.encode(line).ids

                for i in range(0, len(tokens) - self.block_size, self.block_size):
                    x = torch.tensor(
                        tokens[i:i+self.block_size],
                        dtype=torch.long
                    )
                    y = torch.tensor(
                        tokens[i+1:i+self.block_size+1],
                        dtype=torch.long
                    )
                    yield x, y
