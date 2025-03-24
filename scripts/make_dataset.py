from datasets import load_dataset
import torch

class HateSpeechDataset:
    """
    A class to handle all helper functions to load the hate speech
    """
    TEST_DATA_PATH = "./data/processed_data/test.json"
    TRAIN_DATA_PATH = "./data/processed_data/train.json"

    @staticmethod
    def process_data(tokenizer, data_path = TEST_DATA_PATH):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '[PAD]'
        dataset = load_dataset("json", data_files=data_path, split="train[:1500]")

        def format_example(example):
            instruction = example['Question']
            inputs = tokenizer(
                f"{instruction}\n<|endoftext|>",
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(0), 
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": example["Label"]
            }

        return dataset.map(format_example, remove_columns=dataset.column_names)
    
    @staticmethod
    def data_collator(data):
        batch = {
            "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
            "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
            # use input_ids as labels.
            "labels": torch.stack([torch.tensor(d["labels"]) for d in data])
        }
        return batch
    

if __name__ == "__main__":
    from transformers import BertTokenizer
    MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = HateSpeechDataset.process_data(tokenizer)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(test_dataset, batch_size=8, collate_fn = HateSpeechDataset.data_collator)
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']