"""
This module is intended as a more production-thinking demonstration
of how the transformers peft functionality could be exposed behind
a simple REST API. This is not intended as a recommendation for
how to productionalize adapters, but, on the contrary, a demonstration
of the orchestration challenges which could arise when attempting to
naively employ an adapter development library for production inferencing.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = None
tokenizer = None
adapters = {}


class PredictRequest(BaseModel):
    prompt: str
    adapter_name: str


def load_model_and_tokenizer(model_id: str):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to("cuda" if torch.cuda.is_available() else "cpu")


def load_adapters(adapter_configs):
    global model, adapters
    for adapter_name, adapter_path in adapter_configs.items():
        model.load_adapter(adapter_path, adapter_name=adapter_name)
        adapters[adapter_name] = adapter_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # note that we must explicitly load the adapters using a lifespan,
    # which is not only tying our adapter management to global module
    # state but would be difficult to manage if we had more adapters,
    # which could not all be fit into VRAM.
    load_model_and_tokenizer("facebook/opt-350m")
    adapter_configs = {
        "one": "ybelkada/opt-350m-lora",
        "two": "HeydarS/opt-350m-qlora",
    }
    load_adapters(adapter_configs)
    yield


app = FastAPI(lifespan=lifespan)


def generate(prompt: str, adapter_name: str) -> str:
    global model, tokenizer
    if adapter_name not in adapters:
        raise ValueError(f"Adapter '{adapter_name}' is not loaded.")

    # here we must explicitly load the adapter when handling the
    # request. this works, but does not handle unloading
    # or, importantly, batching of requests targeting the same adapter
    model.set_adapter(adapter_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=20)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

    return output


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        output = generate(request.prompt, request.adapter_name)
        return {"generated_text": output}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
