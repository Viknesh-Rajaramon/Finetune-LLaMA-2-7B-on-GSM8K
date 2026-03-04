# LLaMA-2 Fine Tuning on GSM8K with LoRA and 4 Bit Quantization
This project implements parameter-efficient fine-tuning of LLaMA 2 7B on the GSM8K benchmark using LoRA adapters and 4-bit NF4 quantization.
The goal is to enable memory-efficient large language model training while maintaining strong generalization performance on mathematical reasoning tasks.

## Overview
Large language models require significant GPU memory and compute for fine-tuning. This project demonstrates:
- Parameter-efficient fine-tuning using LoRA
- 4-bit quantized model loading using BitsAndBytes
- Supervised fine-tuning with structured Chain of Thought prompting
- Automated validation and experiment tracking
- Evaluation on GSM8K test split with exact match accuracy

The full pipeline is implemented using:
- Hugging Face Transformers
- PEFT
- TRL
- Weights & Biases
