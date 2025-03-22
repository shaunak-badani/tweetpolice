import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import numpy as np

# Configurations
model_path = "distilbert/distilgpt2"
data_path = "../data/processed_data/train.json"
output_path = "../output"

assert torch.cuda.is_available(), "Use GPU!"
device = torch.device("cuda")

class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])

def process_data(tokenizer):
    dataset = load_dataset("json", data_files=data_path, split="train[:1500]")
    print(dataset)

    def format_example(example):
        # Include the label vector in the instruction
        vector_str = f"Label Distribution: [hate speech: {example['LabelVector'][0]:.2f}, offensive language: {example['LabelVector'][1]:.2f}, neither: {example['LabelVector'][2]:.2f}]"
        instruction = f"Question: {example['Question']}\n{vector_str}\nAnalysis: {example['Complex_CoT']}"
        inputs = tokenizer(
            f"{instruction}\n### Answer: \n{example['Response']}<|endoftext|>",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0)}

    return dataset.map(format_example, remove_columns=dataset.column_names)

# Add gradient clipping
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],  # Added c_fc as target
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  # Reduced from 500 to a more reasonable number
    learning_rate=3e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="steps",  # Changed from "no" to "steps"
    save_steps=100,
    eval_steps=100,  # Added evaluation steps
    max_grad_norm=1.0,  # Added gradient clipping
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",  # Added learning rate scheduler
    warmup_ratio=0.1,  # Added warmup
    no_cuda=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False
)

def main():
    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and split dataset for training and evaluation
    full_dataset = load_dataset("json", data_files=data_path)
    dataset_dict = full_dataset.train_test_split(test_size=0.1)
    
    train_dataset = process_data(tokenizer)
    eval_dataset = dataset_dict["test"].map(
        lambda x: process_data(tokenizer)._data[0],  # Use the same processing for evaluation
        batched=False
    )
    
    loss_callback = LossCallback()

    def data_collator(data):
        batch = {
            "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device),
            "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]).to(device),
            "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device)
        }
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Added evaluation dataset
        data_collator=data_collator,
        callbacks=[loss_callback]
    )

    print("Start training...")
    trainer.train()

    trainer.model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    print("Loss curve saved to loss_curve.png")

if __name__ == "__main__":
    main()
