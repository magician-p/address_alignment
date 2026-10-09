import json
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

from address_alignment import address_alignment
from config import *

app = FastAPI()

model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_PATH if pretrained_exists() else PRETRAINED_MODEL_NAME
)
with open(PROCESSED_DATA_PATH / 'id2label.json', 'r', encoding='utf-8') as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

class MessageRequest(BaseModel):
    message : str = ''

class MessageResponse(BaseModel):
    Prov: str | None = None
    City: str | None = None
    District: str | None = None
    Town: str | None = None
    Detail: str | None = None
    Name: str | None = None
    Phone: str | None = None

@app.get('/')
async def homepage():
    return FileResponse(PROJECT_ROOT / 'index.html')

@app.post('/address_alignment')
async def handle_message(request: MessageRequest):
    addresses = address_alignment([request.message], model, tokenizer, id2label)
    if not addresses:
        return MessageResponse()
    return MessageResponse(**asdict(addresses[0]))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8089)
