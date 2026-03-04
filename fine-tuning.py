import os

# Set Environment Variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import wandb

import torch
from datetime import datetime


class ModelConfig():
	def __init__(self):
		self.model_name = "meta-llama/Llama-2-7b-hf" # Model name
		self.wandb_api_key = ""  # Weights & Biases API Key
		self.hf_token = "" # Hugging Face Token
		self.dataset_name = "openai/gsm8k" # Dataset Name
		self.test_split = 0.1  # Split 10% for validation
		self.output_dir = "" # Directory to store the results/output
		self.num_epochs = 5 # Number of epochs
		self.train_batch_size = 20 # Training batch size
		self.val_batch_size = 20 # Validation batch size
		self.gradient_accumulation_steps = 10 # Gradient accumulation steps
		self.dataloader_num_workers = 4 # Number of dataloader workers (Hugging Face)
		self.val_strategy = "steps" # Validation strategy
		self.val_steps = 20 # Validation steps
		self.logging_strategy = "steps" # Logging strategy
		self.logging_steps = 1 # Logging steps
		self.optimizer = "paged_adamw_8bit" # Optimizer
		self.learning_rate = 1e-4 # Learning rate
		self.lr_scheduler_type = "linear" # Learning rate scheduler type
		self.warmup_steps = 10 # Warm up steps
		self.report_to = "wandb"
		self.max_seq_length = 512 # Maximum sequence length
		self.save_model_dir = "" # Directory to save the model being fine-tuned
		self.wandb_user = "" # Username for Weight & Biases


def wandb_login(api_key, user):
    wandb.login(key = api_key)
    wandb.init(project = f"Llama-2-7b-hf-{user}-{datetime.now().strftime('%Y-%m-%d')}")


def get_model_config():
	model_config = ModelConfig()
	return model_config


def load_model_and_tokenizer(model_name, hf_token):
	bnb_config = BitsAndBytesConfig(
		load_in_4bit = True,
		bnb_4bit_quant_type = "nf4",
		bnb_4bit_compute_dtype = torch.float16,
		bnb_4bit_use_double_quant = True,
	)

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		quantization_config = bnb_config,
		device_map = {"": 0},
		use_cache = False,
	)
	
	model = prepare_model_for_kbit_training(model)
	
	tokenizer = AutoTokenizer.from_pretrained(
		pretrained_model_name_or_path = model_name,
		token = hf_token,
	)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	
	return model, tokenizer


def get_preprocessed_dataset(train_dataset, test_split):
	train_dataset = train_dataset

	def chat_template(example):
		example["question"] = "Question:\n {question} \n\n Let's think step by step. At the end, you MUST write the answer as an integer after '####'.\n\n Answer:\n {answer}".format(question = example["question"], answer = example["answer"])
		return example

	# Split the training dataset into training and validation sets
	train_dataset, val_dataset = train_dataset.train_test_split(test_size = test_split).values()

	train_dataset = train_dataset.map(chat_template, remove_columns = ["answer"])
	val_dataset = val_dataset.map(chat_template, remove_columns = ["answer"])

	return train_dataset, val_dataset


def get_dataset(dataset_name, hf_token, test_split):
	train_dataset = load_dataset(
		path = dataset_name,
		token = hf_token,
		name = "main",
		split = "train"
	)

	train_dataset, val_dataset = get_preprocessed_dataset(train_dataset, test_split)

	return train_dataset, val_dataset


def get_peft_config():
	config = LoraConfig(
		r = 16,
		lora_alpha = 32,
		lora_dropout = 0.05,
		bias = "none",
		task_type = "CAUSAL_LM",
		target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	)

	return config


def get_sft_config(model_config):
	config = SFTConfig(
		output_dir = model_config.output_dir,
		num_train_epochs = model_config.num_epochs,
		per_device_train_batch_size = model_config.train_batch_size,
		per_device_eval_batch_size = model_config.val_batch_size,
		dataloader_num_workers = model_config.dataloader_num_workers,
		gradient_accumulation_steps = model_config.gradient_accumulation_steps,
		eval_strategy = model_config.val_strategy,
		eval_steps = model_config.val_steps,
        do_eval = True,
		logging_strategy = model_config.logging_strategy,
		logging_steps = model_config.logging_steps,
		optim = model_config.optimizer,
		learning_rate = model_config.learning_rate,
		lr_scheduler_type = model_config.lr_scheduler_type,
		warmup_steps = model_config.warmup_steps,
		report_to = model_config.report_to,
		max_seq_length = model_config.max_seq_length,
		dataset_text_field = "question",
        save_strategy = "steps",
	)

	return config


def get_sft_trainer(model, train_dataset, val_dataset, peft_config, tokenizer, sft_config):
	trainer = SFTTrainer(
		model = model,
		train_dataset = train_dataset,
		eval_dataset = val_dataset,
		peft_config = peft_config,
		tokenizer = tokenizer,
		args = sft_config,
	)

	return trainer


def main():
	# Get model config
	model_config = get_model_config()

	# Check if CUDA is available
	print(f"\nCUDA available: {torch.cuda.is_available()}")

	# Login to WandB for logging curves
	wandb_login(model_config.wandb_api_key, model_config.wandb_user)

	# Load model and tokenizer
	model, tokenizer = load_model_and_tokenizer(model_config.model_name, model_config.hf_token)
	
	# Load preprocessed dataset
	train_dataset, val_dataset = get_dataset(model_config.dataset_name, model_config.hf_token, model_config.test_split)

	print(f"\nTraining Dataset Length = {len(train_dataset)}")
	print(f"\nValidation Dataset Length = {len(val_dataset)}")

	# Get PEFT parameters
	peft_config = get_peft_config()

	# Get SFT parameters. Used instead of TrainingArguments (deprecated)
	sft_config = get_sft_config(model_config)

	# Get SFT trainer
	trainer = get_sft_trainer(model, train_dataset, val_dataset, peft_config, tokenizer, sft_config)

	# Train the model
	trainer.train()

	# Save the trained model
	trainer.model.save_pretrained(save_directory = model_config.save_model_dir)

	# Close WandB connection
	wandb.finish()
	


if __name__ == "__main__":
	main()
