import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from webdojo.agent.prompt import SYSTEM_PROMPT


class WebDojo_Agent_Chat:
    
	def __init__(self, model_name: str, system_prompt: str = "default"):
		if "glm" in model_name:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
			self.model = AutoModelForCausalLM.from_pretrained(
				model_name, 
				torch_dtype=torch.float16,
				device_map="auto",
				trust_remote_code=True
			)
		else:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name)
			self.model = AutoModelForCausalLM.from_pretrained(
				model_name, 
				torch_dtype=torch.float16,
				device_map="auto"
			)

		self.goal = None
		self.action_history = []
  
  
	def reset_agent(self, new_goal):
		self.goal = new_goal
		self.action_history = []
		self.conversations = []

	def get_response(self, html: str, prefix: str = "") -> str:
		
		conversations = self.get_chat_prompt(html)
			
		prompt = self.tokenizer.apply_chat_template(
			conversations, 
			tokenize=False,
			add_generation_prompt=True
		)
		if prefix:
			prompt = prompt + prefix
  
		# encode prompt
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
  
		# generate response
		with torch.no_grad():
			output = self.model.generate(**inputs, 
                                max_new_tokens=512,
                                do_sample=False,
                                top_p=None,
                                temperature=None,
                                )
		input_length = inputs.input_ids.shape[1]
		generated_tokens = output[0][input_length:]
  
		# decode response
		model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
		if prefix:
			model_output = prefix + model_output
  
		self.action_history.append(model_output)
  
		print("Raw response:", model_output)
  
		return {
			"model_output": model_output,
			"conversations": conversations,
			"user_prompt": self.get_webrl_prompt(html),
		}


	def get_chat_prompt(self, html: str):
  
		index = len(self.action_history)
		conversations = []
		for i in range(index - 1, -1, -1):
			if i == 0:
				content_user = f"Task Instruction: {self.goal}\n\nRound {i}\n{self.goal}"
				content_assistant = f"{self.action_history[i]}"
			else:
				content_user = f"Round {i}\n** Simplified html **"
				content_assistant = f"{self.action_history[i]}"
			conversation = [{'role': 'user', 'content': content_user}, {'role': 'assistant', 'content': content_assistant}]
			conversations = conversation + conversations
   
		if index == 0:
			conversations = [
				{'role': 'system', 'content': SYSTEM_PROMPT},
				{'role': 'user', 'content': f"Task Instruction: {self.goal}\n\nRound {index}\n{html}"}
			]
			return conversations
        
		system_turn = [{'role': 'system', 'content': SYSTEM_PROMPT}]
		current_turn = [{'role': 'user', 'content': f'Round {index}\n\n{html}'}]
		conversations = system_turn + conversations + current_turn
  
		return conversations


	def get_webrl_prompt(self, html: str):
		# set system prompt
		prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>{USER_PROMPT}\n\n"
  
		index = len(self.action_history)
		history = ""
		for i in range(index - 1, -1, -1):
			if i == 0:
				history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{self.goal}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{self.action_history[i]}\n\n" + history
			else:
				history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n** Simplified html **\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{self.action_history[i]}\n\n" + history
		if len(history) + len(html) > (16384 - 512):
			html = html[:(16384 - 512)-len(history)]
		current_turn = f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{html}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
		user_prompt = f"Task Instruction: {self.goal}\n\n{history}{current_turn}"
  
		return user_prompt
