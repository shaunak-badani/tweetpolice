import onnxruntime as ort
from transformers import BertTokenizer

device = "cpu"
ort_session = ort.InferenceSession("models/model.onnx")
input_text = "You look like a hippopotamus!"

MODEL_NAME = "bert-base-uncased" 
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
input_dict = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
dummy_input = {
    "input_ids": input_dict["input_ids"].numpy(),
    "token_type_ids": input_dict["token_type_ids"].numpy(),
    "attention_mask": input_dict["attention_mask"].numpy()
}

print(dummy_input)
outputs = ort_session.run(None, dummy_input)
print(outputs)