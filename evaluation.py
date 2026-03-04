from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from tqdm import tqdm

import torch


class TestConfig():
	def __init__(self):
		self.model_dir = "" # Model directory
		self.hf_token = "" # Hugging Face Token
		self.dataset_name = "openai/gsm8k" # Dataset Name
		self.prompt = "Question:\n {question} \n\n Let's think step by step. At the end, you MUST write the answer as an integer after '####'.\n\n Answer:\n " # Prompt


def get_generation_config(model, tokenizer):
	generation_config = model.generation_config
	generation_config.max_new_tokens = 150
	generation_config.temperature = 0.1
	generation_config.top_p = 0.5
	generation_config.num_return_sequences = 1
	generation_config.pad_token_id = tokenizer.eos_token_id
	generation_config.eos_token_id = tokenizer.eos_token_id

	return generation_config


def get_test_config():
	test_config = TestConfig()
	return test_config


def get_dataset(dataset_name, hf_token):
	dataset = load_dataset(
		path = dataset_name,
		token = hf_token,
		name = "main",
		split = "test"
	)

	return dataset


def load_model_and_tokenizer(model_dir):
	bnb_config = BitsAndBytesConfig(
		load_in_4bit = True,
		bnb_4bit_quant_type = "nf4",
		bnb_4bit_compute_dtype = torch.float16,
		bnb_4bit_use_double_quant = True,
	)
	
	config = PeftConfig.from_pretrained(model_dir)
	model = AutoModelForCausalLM.from_pretrained(
		config.base_model_name_or_path,
		return_dict = True,
		quantization_config = bnb_config,
		device_map = "cuda",
		trust_remote_code = True
	)
	
	tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
	tokenizer.pad_token = tokenizer.eos_token
	
	model = PeftModel.from_pretrained(model, model_dir)
	
	return model, tokenizer


def extract_ans_from_response(answer, eos = None):
	if eos:
		answer = answer.split(eos)[0].strip()

	try:
		answer = answer.split('####')[1].strip()
	except:
		answer = answer.split('####')[-1].strip()

	for remove_char in [',', '$', '%', 'g']:
		answer = answer.replace(remove_char, '')
	
	return answer
	

def get_response(model, tokenizer, prompt): 
	generation_config = get_generation_config(model, tokenizer)

	encoding = tokenizer(prompt, return_tensors = "pt").to("cuda")
	with torch.inference_mode():
		output = model.generate(
			input_ids = encoding.input_ids,
			generation_config = generation_config
		)
	
	decoded_output = tokenizer.decode(output[0], skip_special_tokens = True)
	
	return decoded_output


def run_prediction(model, tokenizer, dataset, prompt_format):
	correct, total = 0, 0

	for qna in tqdm(dataset):
		prompt = prompt_format.format(question = qna["question"])
		response = get_response(model, tokenizer, prompt)
		
		pred_ans = extract_ans_from_response(response[len(prompt) : ])
		true_ans = extract_ans_from_response(qna["answer"])

		total += 1
		if pred_ans == true_ans:
			correct += 1
	
	return correct, total


def test():
	# Check if CUDA is available
	print(f"\nCUDA available: {torch.cuda.is_available()}")

	# Get model config
	test_config = get_test_config()

	# Load dataset
	dataset = get_dataset(test_config.dataset_name, test_config.hf_token)

	# Load trained model and tokenizer
	model, tokenizer = load_model_and_tokenizer(test_config.model_dir)

	# Get model predictions and correct answer for test dataset
	correct_predictions, total_predictions = run_prediction(model, tokenizer, dataset, test_config.prompt)
	
	# Compute test accuracy
	test_acc = correct_predictions / total_predictions * 100

	# Print testing accuracy
	print("Testing Accuracy: {test_acc:.3f} %".format(test_acc = test_acc))



if __name__ == "__main__":
	test()
