from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import torch

class LoraFineTuner:
    BASE_MODEL_OUTPUT_PATH = "./base_output"
    OUTPUT_PATH = "./output"

    @staticmethod
    def _get_config():
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=None
        )
    
    @staticmethod
    def get_lora_wrapped_model(model):
        lora_model = get_peft_model(model, LoraFineTuner._get_config())

        for name, param in lora_model.named_parameters():
            param.requires_grad = "classifier" in name or "lora" in name

        return lora_model
    
    @staticmethod
    def get_training_arguments():
        return TrainingArguments(
            output_dir=LoraFineTuner.OUTPUT_PATH,
            per_device_train_batch_size=4,  # storage limited.
            gradient_accumulation_steps=8,  # accumulate gradient, batch_size=8
            num_train_epochs=0.2,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=500,
            save_strategy="no",
            report_to="none",
            optim="adamw_torch",
            no_cuda=False,
            dataloader_pin_memory=False,  # use pinned memory to accelerate training.
            remove_unused_columns=False  # prevent error.
        )
    


class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs = False):
        """
        Custom loss computation method
        
        Args:
            model: The model
            inputs: Input dictionary
            return_outputs: Whether to return model outputs along with loss
        
        Returns:
            Loss, or (loss, outputs) if return_outputs is True
        """
        # Forward pass to get logits
        outputs = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )
        
        # Extract logits and labels
        logits = outputs.logits
        labels = inputs['labels']

        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        # Compute cross-entropy loss
        loss_value = loss(
            logits,
            labels
        )
        return (loss_value, outputs) if return_outputs else loss_value


class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])

if __name__ == "__main__":
    from transformers import BertTokenizer, BertForSequenceClassification
    from make_dataset import HateSpeechDataset
    MODEL_NAME = "bert-base-uncased" 
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # Adjust num_labels based on the task
    lora_model = LoraFineTuner.get_lora_wrapped_model(model)
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(f"- {name}")

    train_dataset = HateSpeechDataset.process_data(BertTokenizer.from_pretrained(MODEL_NAME), data_path= HateSpeechDataset.TRAIN_DATA_PATH)
    
    trainer = CustomLossTrainer(
        model=lora_model,
        args=LoraFineTuner.get_training_arguments(),
        train_dataset=train_dataset,
        data_collator=HateSpeechDataset.data_collator,
        callbacks=[LossCallback()],
    )