### Tips for Serving Language Model Adapters at Scale
Language Models have become very popular in recent times and adapters are a powerful technology for allowing adaptation of model behaviour with a small number of parameters. In order to make effective use of language model adapters in production environments, there are several tips worth mentioning to ensure scalable serving and effective management of base models and adapters.

This directory contains the original presentation slides from the talk, as well as code for the accompanying demonstrations.

#### Getting Started (Code)
1. Install PyTorch according to the instructions [here](https://pytorch.org/get-started/locally/).
2. Install requirements.txt.
3. Run demo.py

#### Additional Notes
- Be cautious regarding adapters loaded from Hugging Face repositories. You should only require a .safetensors and adapter_config.json file, as in the adapters used in this demonstration.

***
