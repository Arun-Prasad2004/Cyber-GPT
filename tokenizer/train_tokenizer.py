from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

corpus_path = "data/processed/shuffled_corpus.txt"
save_path = "tokenizer/cyber_tokenizer.json"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]"]
)

tokenizer.train([corpus_path], trainer)
tokenizer.save(save_path)

print("Tokenizer trained and saved.")
