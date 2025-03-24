from transformers import BertTokenizer, BertForSequenceClassification
from scripts.make_dataset import HateSpeechDataset
from scripts.fine_tune import LoraFineTuner, CustomLossTrainer, LossCallback


def main():
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
    trainer.train()

main()