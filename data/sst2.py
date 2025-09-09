from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader

def get_sst2(batch_size=32):
    dataset = load_dataset("glue", "sst2")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return (
        DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True),
        DataLoader(tokenized["validation"], batch_size=batch_size),
    )