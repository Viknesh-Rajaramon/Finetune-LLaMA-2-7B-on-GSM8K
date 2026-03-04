# LLaMA-2 Fine Tuning on GSM8K with LoRA and 4 Bit Quantization
This project implements parameter-efficient fine-tuning of LLaMA 2 7B on the GSM8K benchmark using LoRA adapters and 4-bit NF4 quantization.
The goal is to enable memory-efficient training of large language models while maintaining strong generalization performance on mathematical reasoning tasks.

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

## Architecture
### Model
- Base model: LLaMA 2 7B
- Task type: Causal Language Modeling
- LoRA rank: 16
- LoRA alpha: 32
- Dropout: 0.05
- Target modules: attention and MLP projection layers

### Quantization
- 4-bit NF4 quantization
- Float16 compute dtype
- Double quantization enabled
- Prepared for k-bit training

This setup enables training a 7B-parameter model on a single GPU with a significantly reduced memory footprint.

## Dataset
- Dataset: GSM8K
- Split: 90 percent train, 10 percent validation
- Test split used for final evaluation
- Chain of Thought prompting format enforced
- Final answer extraction using structured delimiter ####

Prompt format:
```
Question:
{question}

Let's think step by step. At the end, you MUST write the answer as an integer after '####'.

Answer:
```

Training Configuration

Epochs: 5

Train batch size: 20

Gradient accumulation steps: 10

Effective batch size: 200

Optimizer: paged_adamw_8bit

Learning rate: 1e-4

Scheduler: Linear

Warmup steps: 10

Max sequence length: 512

Validation strategy: Step based

Logging: Every step to Weights and Biases

Results
Convergence Behavior

Training loss reduced from 1.63 to 0.67

Validation loss stabilized near 0.70

Training and validation curves closely aligned

No significant overfitting observed

The model demonstrated stable convergence and strong generalization on unseen GSM8K test samples.

Evaluation

The evaluation script:

Loads quantized base model

Applies trained LoRA adapters

Generates responses with controlled decoding

Extracts answers using delimiter parsing

Computes exact match accuracy

Inference configuration:

Max new tokens: 150

Temperature: 0.1

Top p: 0.5

Project Structure
.
├── fine_tuning.py
├── evaluation.py
├── results/
└── README.md

fine_tuning.py: Training pipeline

evaluation.py: Test time inference and accuracy computation

results/: Model checkpoints and logs

Key Learnings

LoRA significantly reduces trainable parameters while preserving model quality

4 bit quantization enables large model training on limited hardware

Step based validation provides early detection of overfitting

Structured prompting improves reasoning consistency on math datasets

Future Improvements

Hyperparameter sweep using Weights and Biases

Compare LoRA ranks and dropout configurations

Add instruction tuning datasets for broader reasoning capability

Benchmark against non quantized baseline
