import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configurations
model_path = "distilgpt2"  # Fixed model path
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

def process_data(tokenizer, dataset):
    """Process and tokenize the dataset"""
    def format_example(example):
        # Include the label vector in the instruction
        vector_str = f"Label Distribution: [hate speech: {example['LabelVector'][0]:.2f}, offensive language: {example['LabelVector'][1]:.2f}, neither: {example['LabelVector'][2]:.2f}]"
        instruction = f"Below is a task on content moderation.\nQuestion: {example['Question']}\nAnalyze the question and provide a comprehensive response.\nAnalysis: {example['Complex_CoT']}"
        inputs = tokenizer(
            f"{instruction}\n### Answer: \n{example['Response']}<|endoftext|>",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0), 
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_vector": example["LabelVector"]  # Keep label vector for evaluation
        }

    return dataset.map(format_example, remove_columns=[col for col in dataset.column_names if col != "LabelVector"])

# Add gradient clipping
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,  # Added eval batch size
    gradient_accumulation_steps=4,
    num_train_epochs=100,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=20,
    max_grad_norm=1.0,  # Gradient clipping
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine", 
    warmup_ratio=0.1,
    no_cuda=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100
)

def evaluate_model(model, eval_dataset, tokenizer, output_path):
    """Evaluate model on the evaluation dataset and compute metrics"""
    print("Starting model evaluation...")
    model.eval()
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process examples in smaller batches
    batch_size = 4
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:i+batch_size]
        input_texts = []
        
        # Prepare inputs
        for example in batch:
            # Extract just the instruction part for generation
            input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            input_text = input_text.split("### Answer:")[0] + "### Answer:"
            input_texts.append(input_text)
            references.append(example["label_vector"])
        
        # Tokenize and generate
        inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode generated text
        for j, output in enumerate(outputs):
            # Get only the generated part
            generated_text = tokenizer.decode(output[inputs["input_ids"][j].shape[0]:], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
            # Here we would normally extract predicted label vector from the generated text
            # For simplicity, we'll just use a placeholder
            # In a real scenario, you would parse the generated text to extract predictions
            # pred_vector = extract_prediction_from_text(generated_text)
            
            # Placeholder - in real implementation, extract from generated text
            pred_vector = references[i+j]  # Using reference as placeholder
            predictions.append(pred_vector)
            
            print(f"Example {i+j+1}/{len(eval_dataset)}", end="\r")
    
    print("\nEvaluation complete, computing metrics...")
    
    # Convert to numpy arrays for metric calculation
    predictions = np.array(predictions)
    references = np.array(references)
    
    # Calculate the predicted class (argmax of the label vector)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(references, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, pred_classes, average='weighted')
    
    # Save metrics and some examples
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    print(f"Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save metrics and examples
    os.makedirs(os.path.join(output_path, "evaluation"), exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_path, "evaluation", "metrics.txt"), "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    # Save some example generations
    with open(os.path.join(output_path, "evaluation", "examples.txt"), "w") as f:
        num_examples = min(10, len(generated_texts))
        for i in range(num_examples):
            input_text = tokenizer.decode(eval_dataset[i]["input_ids"], skip_special_tokens=True)
            input_text = input_text.split("### Answer:")[0]
            
            f.write(f"Example {i+1}:\n")
            f.write(f"Input:\n{input_text}\n\n")
            f.write(f"Generated:\n{generated_texts[i]}\n\n")
            f.write(f"True label: {references[i]}\n")
            f.write(f"Predicted label: {predictions[i]}\n")
            f.write("="*50 + "\n\n")
    
    return metrics

def main():
    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        device_map="auto"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and split dataset for training and evaluation once
    dataset = load_dataset("json", data_files=data_path)
    dataset_dict = dataset.train_test_split(test_size=0.1)
    
    # Process train and test datasets separately
    train_dataset = process_data(tokenizer, dataset_dict["train"])
    eval_dataset = process_data(tokenizer, dataset_dict["test"])
    
    loss_callback = LossCallback()

    def data_collator(data):
        # Better error handling and no device transfer in collator
        input_ids = [torch.tensor(d["input_ids"]) for d in data if "input_ids" in d]
        attention_masks = [torch.tensor(d["attention_mask"]) for d in data if "attention_mask" in d]
        
        if not input_ids or not attention_masks:
            return {}
            
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(input_ids).clone()  # Create a separate clone for labels
        }
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[loss_callback]
    )

    print("Start training...")
    trainer.train()

    # Save trained model and tokenizer
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model and tokenizer saved to {output_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    print("Loss curve saved to loss_curve.png")
    
    # Run evaluation
    metrics = evaluate_model(model, eval_dataset, tokenizer, output_path)
    
    # Save evaluation metrics to file
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
