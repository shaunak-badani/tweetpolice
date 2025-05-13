import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.onnx
from fine_tune import LoraFineTuner
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import os

# SELECT DEVICE TO RUN INFERENCE ON
device = "cpu"

MODEL_NAME = "bert-base-uncased" 
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # Adjust num_labels based on the task
# lora_model = LoraFineTuner.get_lora_wrapped_model(model)
lora_model = PeftModel.from_pretrained(model, "./output")
lora_model.to(device)
lora_model.eval()

dummy_input_text = "Please classify the sentiment of the following sentence as Positive, Negative, or Neutral: 'I love this product!'"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
dummy_input = tokenizer(dummy_input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
print(dummy_input)


os.makedirs("models", exist_ok=True)
torch.onnx.export(model, (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids']), "models/model.onnx", input_names = ['input_ids', 'attention_mask', 'token_type_ids'], output_names = ['op1'], verbose = True, dynamic_axes={
    'input_ids': {0: 'batch_size', 1: 'seq_length'},
    'attention_mask': {0: 'batch_size', 1: 'seq_length'},
    'output': {0: 'batch_size', 1: 'seq_length'}
})