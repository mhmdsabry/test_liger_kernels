import configparser
import argparse 
import os
import gc

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from utils import *
from prepare_dataset import DatasetConfig, Alpaca, ARC
from finetuner import FintunerConfig, Finetuner

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Linger Tests")
parser.add_argument("-c","--config",dest="filename", help="Pass a training config file",metavar="FILE")
parser.add_argument("--exp_name", help="Experiment name",type=str, default="test_linger_kernels")
parser.add_argument("--model_name", help="Model name (Huggingface Hub model name)",type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--dataset_name", help="Dataset name (Huggingface Hub data name)",type=str, default="tatsu-lab/alpaca")
parser.add_argument("--learning_steps", help="Training Steps", type=int, default=500)
parser.add_argument("--max_seq_length", help="Sequence tokens length", type=int, default=1024)
parser.add_argument("--train_batch_size", help="Training batch size", type=int, default=8)
parser.add_argument("--logging_steps", help="Training Logging steps, set to 1 if you want to track the efficiency of training", type=int, default=1)
parser.add_argument("--include_num_input_tokens_seen", help="Enable to track the efficiency of training", action="store_true", required=False)
parser.add_argument("--use_liger", help="Whether to use Liger Kernels or Not", action="store_true", required=False)
parser.add_argument("--seed", help="Random seed for experiment reproduction",type=int, default=42)


args = parser.parse_args()
config.read(args.filename)

seed = args.seed

#imported from utils
seed_everything(seed)


##################################################################
#                        Set up Configs                          #
##################################################################

                        #################
                        #  exp artifact #
                        #################
exp_name = args.exp_name
resume_exp = config['exp_config']['resume_exp']
exp_folder = config['exp_config']['exp_folder']
cache_dir = config['exp_config']['cache_dir']

outfolder, cache_dir = Path(exp_folder), Path(cache_dir)
outfolder.mkdir(exist_ok=resume_exp)
cache_dir.mkdir(exist_ok=resume_exp)
os.environ['WANDB_PROJECT'] = config['exp_config']['wandb_project_name']

# use to save files, name wandb run, ect..
save_model_name = args.model_name.replace("/","_")
save_dataset_name = args.dataset_name.replace("/","_")

exps_save_path = outfolder / exp_name / f'liger_{save_dataset_name}' if args.use_liger else outfolder / exp_name / f'no_liger_{save_dataset_name}'

                        #################
                        #    training   #
                        #################
task_do_train = config.getboolean('task_config', 'task_do_train')
task_do_eval = config.getboolean('task_config', 'task_do_eval')

# We want to push across train batch size, that's why we made it a CLI arg, other than that,
# just define it in config.ini file
task_per_device_train_batch_size = args.train_batch_size
#task_per_device_train_batch_size = int(config['task_config']['task_per_device_train_batch_size'])


task_per_device_eval_batch_size = int(config['task_config']['task_per_device_eval_batch_size'])
task_learning_rate = float(config['task_config']['task_learning_rate'])
task_gradient_accumulation_steps = int(config['task_config']['task_gradient_accumulation_steps'])
task_gradient_checkpointing = config.getboolean('task_config', 'task_gradient_checkpointing')
task_weight_decay = float(config['task_config']['task_weight_decay'])

task_max_steps = args.learning_steps           
task_warmup_steps = int(0.1 * task_max_steps)  
task_save_steps = int(0.5 * task_max_steps)    
task_eval_steps = int(0.5 * task_max_steps)


                        #################
                        # LoRA modules  #
                        #################
lora_r = int(config['modules_config']['lora_rank'])
lora_alpha = int(config['modules_config']['lora_alpha'])
lora_dropout = float(config['modules_config']['lora_dropout'])


##################################################################
#                       Model Setup                              #
##################################################################

# for llama, defualt optimised kernels are: Rope, RMSnorm, SwisGLU, and either enable cross entropy or fused but not both
tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        truncation_side="left",
    )
tokenizer.pad_token = tokenizer.eos_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"



if args.use_liger:
    model = AutoLigerKernelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch.bfloat16,
                cross_entropy=True,
                fused_linear_cross_entropy=False,
            )
else:
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

lora_model = get_peft_model(model, lora_config)

##################################################################
#                       Dataset Setup                            #
##################################################################


dataconfig = DatasetConfig(
    dataset_name = args.dataset_name,
    tokenizer = tokenizer,
    )


if "alpaca" in save_dataset_name.lower():
    alpaca = Alpaca(dataconfig)
    finetuneset, evalset, formatting_prompts_func, collator = alpaca.preprocess()
elif "commonsense" in save_dataset_name.lower():
    arc = ARC(dataconfig)
    finetuneset, evalset, formatting_prompts_func, collator = arc.preprocess()
else:
    raise ValueError(f"Preprocessing steps not implemented for dataset: {args.dataset_name}")


##################################################################
#                  LoRA-Finetuning                               #
##################################################################

task_finetune_config = FintunerConfig(
            output_dir = exps_save_path,
            do_train = task_do_train,
            do_eval = task_do_eval,
            per_device_train_batch_size = task_per_device_train_batch_size,
            per_device_eval_batch_size = task_per_device_eval_batch_size,
            learning_rate = task_learning_rate,
            max_steps = task_max_steps,
            warmup_steps = task_warmup_steps,
            save_steps = task_save_steps,
            eval_steps = task_eval_steps,
            gradient_accumulation_steps = task_gradient_accumulation_steps,
            gradient_checkpointing = task_gradient_checkpointing,
            run_name = f"{exp_name}_{'liger' if args.use_liger else 'no_liger'}_{save_model_name}_{save_dataset_name}",
            data_collator = collator,
            weight_decay=task_weight_decay,            
            seed = seed,
            include_num_input_tokens_seen=args.include_num_input_tokens_seen,
            logging_steps=args.logging_steps,
            formatting_prompts_func = formatting_prompts_func,
            max_seq_length = args.max_seq_length
            )


if __name__ == "__main__":
    finetuner = Finetuner(lora_model, tokenizer, finetuneset, evalset, task_finetune_config)
    finetuner.finetune()
    lora_model.save_pretrained(exps_save_path)

    #if "alpaca" in args.dataset_name.lower() and args.use_liger:
    #    lora_model.push_to_hub("linger_llama_lora_alpaca")
    #elif "commonsense" in args.dataset_name.lower() and args.use_liger:
    #    lora_model.push_to_hub("linger_llama_lora_arc")


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()