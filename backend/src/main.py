from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


# Inference libraries
import onnxruntime as ort
from transformers import BertTokenizer
import numpy as np

app = FastAPI(root_path='/api')

# list of allowed origins
origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get-sentiment")
def get_sentiment_for_prompt(input_text: str):
    """
    Query endpoint to get sentiment for query prompt
    """
    ort_session = ort.InferenceSession("models/model.onnx")
    MODEL_NAME = "bert-base-uncased" 
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)   
    input_dict = tokenizer(input_text, return_tensors="np", truncation=True, padding="max_length", max_length=512)
    input_values = {
        "input_ids": input_dict["input_ids"],
        "token_type_ids": input_dict["token_type_ids"],
        "attention_mask": input_dict["attention_mask"]
    }
    outputs = ort_session.run(None, input_values)[0]
    outputs = outputs - outputs.max()
    percentages = np.exp(outputs) 
    percentages /= percentages.sum() 
    percentages *= 100
    keys = ["Hateful", "Offensive", "Neither hateful nor offensive"]
    prediction = dict(zip(keys, map(float, percentages.flatten())))
    return JSONResponse(
        content = prediction
    )