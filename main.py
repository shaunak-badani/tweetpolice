from transformers import BertTokenizer, BertForSequenceClassification
from scripts.make_dataset import HateSpeechDataset
from scripts.fine_tune import LoraFineTuner, CustomLossTrainer, LossCallback
from scripts.evaluator import Evaluator
import os


def main():
    MODEL_NAME = "bert-base-uncased" 
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # Adjust num_labels based on the task
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Evaluate base model
    os.makedirs("evaluations", exist_ok = True)
    Evaluator.evaluate_model(model, tokenizer, "evaluations/base_model_stats.png")

    # Set up fine tuning
    lora_model = LoraFineTuner.get_lora_wrapped_model(model)
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(f"- {name}")

    train_dataset = HateSpeechDataset.process_data(tokenizer, data_path= HateSpeechDataset.TRAIN_DATA_PATH)

    trainer = CustomLossTrainer(
        model=lora_model,
        args=LoraFineTuner.get_training_arguments(),
        train_dataset=train_dataset,
        data_collator=HateSpeechDataset.data_collator,
        callbacks=[LossCallback()],
    )
    # Run fine tuning
    trainer.train()

    # Evaluate fine tuned model
    Evaluator.evaluate_model(lora_model, tokenizer, "evaluations/fine_tuned_model_stats.png")


main()