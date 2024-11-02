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
from typing import Union

model = None
tokenizer = None
adapters = {}


class PredictRequest(BaseModel):
    prompt: str
    adapter_name: Union[str, None]


def load_model_and_tokenizer(model_id: str):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)


# noinspection PyUnresolvedReferences
def load_adapters(adapter_configs):
    global model, adapters
    for adapter_name, adapter_path in adapter_configs.items():
        model.load_adapter(adapter_path, adapter_name=adapter_name)
        adapters[adapter_name] = adapter_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # note that we must explicitly load the adapters using a lifespan,
    # which is not only tying our adapter management to global
    # state but would be difficult to manage if we had more adapters,
    # which could not all be fit into VRAM.
    load_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")
    adapter_configs = {
        "one": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"
    }
    load_adapters(adapter_configs)
    yield


app = FastAPI(lifespan=lifespan)


# noinspection PyUnresolvedReferences,PyCallingNonCallable
def generate(prompt: str, adapter_name: Union[str, None]) -> str:
    global model, tokenizer

    # here we must explicitly load the adapter when handling the
    # request. this works, but does not handle unloading nicely
    # or, importantly, batching of requests targeting the same adapter
    if adapter_name:
        model.set_adapter(adapter_name)

    else:
        # if a user wants to target the base model without any adapter,
        # we need to explicitly disable the adapters
        model.disable_adapters()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=64)
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
