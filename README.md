# test_liger_kernels
Benchmarking the performance of Liger Kernels Library Using instruction following and reasoning tasks

## Code Structure
```
test_liger_kernels/
├── config.ini: contrain training, lora, and experiments folder configuration.
├── prepare_dataset.py: contain dataset classes for preprocessing steps for Alpaca and Commonsense QA
├── finetuner.py: contain huggingface trainer, training arguments setter.
├── run_finetune.py: where the action happens, and all files are imported there to start LoRA finetuning with Liger kernels or without.
├── utils.py: contain helper functions, like seed everything.
├── callback.py: taken from https://github.com/linkedin/Liger-Kernel/blob/main/examples/huggingface/callback.py, to track training efficiency (memory and throughput)
├── README.md
└── environment.yml
```

## Run Code:
#### Environment:
```
conda env create environment.yml
conda activate liger
```
#### datasets and model:
```
datasets=('tau/commonsense_qa' 'tatsu-lab/alpaca')
model='meta-llama/Meta-Llama-3-8B-Instruct'
```
#### Run with Liger Kernel:
```
python run_finetune.py -c config.ini --exp_name "$exp_name" --model_name "$model" --dataset_name "$dataset" --learning_steps 1000 --logging_steps 1 --include_num_input_tokens_seen --use_liger --train_batch_size "$bs" --max_seq_length "$seqlen"

```
#### Run without Liger Kernel:
```
python run_finetune.py -c config.ini --exp_name "$exp_name" --model_name "$model" --dataset_name "$dataset" --learning_steps 1000 --logging_steps 1 --include_num_input_tokens_seen --train_batch_size "$bs" --max_seq_length "$seqlen"
```
