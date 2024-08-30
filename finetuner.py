import torch

from trl import SFTConfig, SFTTrainer

from callback import EfficiencyCallback


class FintunerConfig:
	def __init__(self,**kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)

class Finetuner(FintunerConfig):
    def __init__(self, model, tokenizer, finetuneset, evalset, finetune_config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.finetuneset = finetuneset
        self.evalset = evalset
        self.formatting_prompts_func = finetune_config.formatting_prompts_func
        self.config = finetune_config

        self.device='cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def finetune(self):
        config = self.config
        training_args = SFTConfig(output_dir=config.output_dir,
                                           overwrite_output_dir = True,
                                           do_train= config.do_train,
                                           do_eval= False if self.evalset==None else config.do_eval,
                                           per_device_train_batch_size=config.per_device_train_batch_size,
                                           per_device_eval_batch_size=config.per_device_eval_batch_size,
                                           max_steps = config.max_steps,
                                           bf16 = True,
                                           max_seq_length=config.max_seq_length,
                                           learning_rate=config.learning_rate,
                                           lr_scheduler_type = "cosine",
                                           warmup_steps=config.warmup_steps,
                                           evaluation_strategy='no' if config.do_eval==False or self.evalset==None else 'steps',
                                           save_strategy= 'steps',
                                           logging_steps = config.logging_steps,
                                           include_num_input_tokens_seen = config.include_num_input_tokens_seen,
                                           seed=config.seed,
                                           save_steps=config.save_steps,
                                           eval_steps=config.eval_steps,
                                           gradient_accumulation_steps=config.gradient_accumulation_steps,
                                           save_total_limit= 1,
                                           dataloader_num_workers = 4,
                                           weight_decay=config.weight_decay,
                                           load_best_model_at_end=False if config.do_eval==False or self.evalset==None else True,
                                           remove_unused_columns = True,
                                           report_to="wandb",
                                           run_name= config.run_name
                                           )
        trainer = SFTTrainer(
                            model=self.model,
                            args=training_args,
                            train_dataset=self.finetuneset,
                            eval_dataset=self.evalset,
                            tokenizer=self.tokenizer,
                            data_collator=config.data_collator,
                            formatting_func= self.formatting_prompts_func,
                            callbacks=[EfficiencyCallback()],
                        )
        
        self.model.print_trainable_parameters()
        trainer.train()

