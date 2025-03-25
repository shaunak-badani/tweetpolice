from scripts.make_dataset import HateSpeechDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

class Evaluator:

    def evaluate_model(model, tokenizer, filename = "test.png"):
        """
        Creates confusion matrices based on the model passed to it
        """
        dataset = HateSpeechDataset.process_data(tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=HateSpeechDataset.data_collator)
        loss = torch.nn.CrossEntropyLoss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        total_cm = 0
        mean_eval_loss = 0

        for data in tqdm(dataloader, desc = "Evaluating..."):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            label = data["labels"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                n = outputs.logits.shape[0]

                # Softmax is applied to the first argument
                pytorch_loss = loss(outputs.logits, label)
                mean_eval_loss += n * pytorch_loss
                pred_classes = torch.argmax(outputs.logits, dim = 1)
                label_classes = torch.argmax(label, dim = 1)
                pred_np = pred_classes.detach().cpu().numpy()
                label_np = label_classes.detach().cpu().numpy()
                cm = confusion_matrix(label_np, pred_np, labels = [0, 1, 2])
                total_cm += cm

        total_cm = total_cm.astype(float)
        total_cm /= total_cm.sum(axis = 1, keepdims = True)
        disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels = ["Hate", "Offensive", "neither"])
        disp.plot()
        plt.savefig(filename)
    
    

if __name__ == "__main__":
    from transformers import BertTokenizer, BertForSequenceClassification
    MODEL_NAME = "bert-base-uncased" 
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # Adjust num_labels based on the task
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    Evaluator.evaluate_model(model, tokenizer, "base_model_stats.png")