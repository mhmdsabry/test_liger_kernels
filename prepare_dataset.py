import datasets
from trl import DataCollatorForCompletionOnlyLM

class DatasetConfig:
	def __init__(self,**kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)

class Alpaca:
	def __init__(self, dataset_config):
		self.config = dataset_config
		self.tokenizer = self.config.tokenizer
		self.dataset_name = self.config.dataset_name
		
		self.dataset = datasets.load_dataset(self.dataset_name)["train"].train_test_split(
            test_size=0.1
            )
	def preprocess(self):
		def formatting_prompts_func(example):
			return example["text"]
		
		train_dataset = self.dataset["train"]
		eval_dataset = self.dataset["test"]
		response_prompt = self.tokenizer.encode("### Response:\n", add_special_tokens=False)
		
		collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template=response_prompt,
                pad_to_multiple_of=16,
        )
		
		return train_dataset, eval_dataset, formatting_prompts_func, collator
	
class ARC:
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.tokenizer = self.config.tokenizer
        self.dataset_name = self.config.dataset_name
		
        self.dataset = datasets.load_dataset(self.dataset_name)
		
    def print_tokens_with_ids(self, txt):
        tokens = self.tokenizer.tokenize(txt, add_special_tokens=False)
        token_ids = self.tokenizer.encode(txt, add_special_tokens=False)
        print(list(zip(tokens, token_ids)))
		
    def preprocess(self):
        def formatting_prompts_func(examples):
            instruction = "Read the question carefully and choose the correct answer by selecting the letter (A, B, C, D, or E) corresponding to the best option. \n ### Question:\n"
            output_text = []
            for i in range(len(examples["question"])):
                question = instruction+examples["question"][i]
                choices = examples["choices"][i]
                reformat_choices = (lambda data: "\n".join([f"{label}) {text}" for label, text in zip(data["label"], data["text"])]))(choices)
               
                answer_key = examples["answerKey"][i]
			    
                text = f"{question}\n### Choices:\n{reformat_choices}\n ### Answer: {answer_key}"
                #self.print_tokens_with_ids(text)
                #self.print_tokens_with_ids("\n### Answer:\n")
                output_text.append(text)
            return output_text
	
        train_dataset = self.dataset["train"]
        eval_dataset = self.dataset["validation"]
		
        #response_prompt = self.tokenizer.encode("\n### Answer:\n", add_special_tokens=False)[1:]
		
        collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template= " ### Answer:",#response_prompt,
                pad_to_multiple_of=16,
        )
		
        return train_dataset, eval_dataset, formatting_prompts_func, collator