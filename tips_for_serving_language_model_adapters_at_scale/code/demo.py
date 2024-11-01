"""
This script implements a simple demonstration of the
adapter loading functionality provided in later versions of
the transformers library. We here demonstrate the ability
to load multiple local adapters and set them on the base
model for generation.
"""
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


def _load_model_and_tokenizer(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return tokenizer, model


# noinspection PyUnresolvedReferences,PyCallingNonCallable
def _generate(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=20)

    outputs = tokenizer.batch_decode(generate_ids)

    return outputs[0]


# noinspection PyUnresolvedReferences
def main() -> None:
    tokenizer, model = _load_model_and_tokenizer("facebook/opt-350m")

    model.load_adapter("ybelkada/opt-350m-lora", adapter_name="one")
    model.load_adapter("HeydarS/opt-350m-qlora", adapter_name="two")

    prompt = "TalkMLOps is"

    model.set_adapter("one")
    print(_generate(prompt, model=model, tokenizer=tokenizer))

    model.set_adapter("two")
    print(_generate(prompt, model=model, tokenizer=tokenizer))

    return None


if __name__ == "__main__":
    main()
