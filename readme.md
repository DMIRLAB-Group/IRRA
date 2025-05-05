# Installation

The code is tested under python=3.10

`pip install -r irra_requirements.txt`
# Preparation

## Dataset

IEPile: https://github.com/zjunlp/IEPile

CrossNER: Provide post-processing

Database：https://drive.google.com/drive/folders/1xDAaTwruESNmleuIsln7IaNleNsYlHGn

## LLaMA-Factory
url: https://github.com/hiyouga/LLaMA-Factory/tree/main

## BCEmbedding And Reranker
url: https://github.com/netease-youdao/BCEmbedding

# Project Structure
We provide a CLI script to easily get the data for each step
```
data
├──database
│  ├──chroma-{chunksize}
|  ├──docs # Unannotated data for different domains
|  |  ├──ai.txt
|  |  ├──....txt
|  |  └──science.txt
|  └──prompts # Prompts for different models and domains
|     └──{model_name}
|        └──docs # Based on retrieval documents
|           ├──ai.md
|           ├──....md
|           └──science.md
├──test
|  ├──crossner
|  |  ├──base # Added instruction
|  |  |  ├──cross.json
|  |  |  ├──ai.json
|  |  |  ├──....json
|  |  |  └──science.json
|  |  └──raw # Original file
|  |     ├──ai.json
|  |     ├──....json
|  |     └──science.json
|  ├──crossner-ec # Added expansion extraction instruction
|  |  └──{model_name}
|  |     ├──cross.ec.json
|  |     └──cross.base.jsonl # results of base extractor
|  ├──crossner-tc # # Added type correction instruction
|  |  └──{model_name}
|  |     ├──cross.tc.{chunksize}.{top_k}.json
|  |     └──cross.ec.jsonl # results of expansion complementor
|  └──crossner-results
|  |  └──{model_name}
|  |     └──cross.tc.{chunksize}.{top_k}.jsonl # final results
└──train
    ├──iepile
    |  └──train.ner.jsonl
    ├──iepile-augmentation
    |  └──train.ner.json
    └──iepile-ec
       └──{model_name}
          ├──train.base.jsonl
          └──train.ec.json
home
└──history # CLI execution history
models
├──bce-embedding # bce-embedding folder
└──bce-reranker # bce-reranker
└──trained_llm # trained llm
src # Script files
run.py  # run this via CLI!
```

# Build Database
Run the CLI and enter `build_databse`.

# Training
## base extractor
For the base extractor, we directly incorporate the trained weights provided by iepile

<!-- > LLaMA-Factory
```
代码
``` -->

## expansion complementor
We first get the augmented iepile dataset
1. Run the CLI and enter  `train` then choose `Get the augmented iepile dataset.` get the `train.ner.json` in `./data/train/iepile-augmentation`
2. Use the following LLaMA-Factory instruction to get the base extractor inference result on `train.ner.json`:
```bash
llamafactory-cli train \
    --model_name_or_path {model_name} \
    --template {model_template} \
    --flash_attn auto \
    --dataset_dir data\train\iepile-augmentation \
    --dataset train.ner \
    --cutoff_len 8192 \
    --max_samples 1000000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 1024 \
    --top_p 0.1 \
    --temperature 0.01 \
    --repetition_penalty 1.1 \
    --output_dir ./data/train/iepile-ec/{model_name} \
    --do_predict True \
    --fp16 True
```
3. Rename the inference result to `train.base.jsonl` and put it in `./data/train/iepile-ec/{model_name}`, then go to the CLI and enter  `train` then choose `Get the extension correction iepile dataset.` (no need to process dev) get the `train.ec.json` in `./data/train/iepile-ec/{model_name}`
4. Use the following LLaMA-Factory instruction, expansion complementor can be obtained by training base extractor with `train.ec.json`
```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path {model_name} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template {model_template} \
    --flash_attn auto \
    --dataset_dir ./data/train/iepile-ec/{model_name} \
    --dataset train.ec \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir models\trained_llm/{model_name/expansion_complementor} \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all 
```
# Testing
1. First you need to get basic crossNER dataset. Run the CLI and enter  `test` then choose `Get the base crossner dataset` (num_schema:3) get the `cross.json` in `./data/test/crossner`
2. Use the following LLaMA-Factory instruction to get the base extractor inference result on `cross.json`:
```bash
llamafactory-cli train \
    --model_name_or_path {model_name} \
    --template {model_template} \
    --flash_attn auto \
    --dataset_dir data\test\crossner\base \
    --dataset cross \
    --cutoff_len 8192 \
    --max_samples 1000000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 1024 \
    --top_p 0.1 \
    --temperature 0.01 \
    --repetition_penalty 1.1 \
    --output_dir ./data/test/crossner-ec/{model_name} \
    --do_predict True \
    --fp16 True
```
3. Rename the inference result to `cross.base.jsonl` and put it in `./data/test/crossner-ec/{model_name}`, then go to the CLI and enter  `test` then choose `Get the extension correction crossner dataset.` get the `cross.ec.json` in `./data/test/crossner-ec/{model_name}`
4. Use the following LLaMA-Factory instruction to get the base extractor inference result on `cross.ec.json`:
```bash
llamafactory-cli train \
    --model_name_or_path {model_name} \
    --template {model_template} \
    --flash_attn auto \
    --dataset_dir ./data/test/crossner-ec/{model_name} \
    --dataset cross.ec \
    --cutoff_len 8192 \
    --max_samples 1000000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 1024 \
    --top_p 0.1 \
    --temperature 0.01 \
    --repetition_penalty 1.1 \
    --output_dir ./data/test/crossner-tc/{model_name} \
    --do_predict True \
    --fp16 True
```
5. Rename the inference result to `cross.ec.jsonl` and put it in `./data/test/crossner-tc/{model_name}`, then go to the CLI and enter  `test` then choose `Get the documents-based correction crossner dataset.` (top_n:20) get the `cross.tc.{chunksize}.{top_k}.json` in `./data/test/crossner-tc/{model_name}`
6. Use the following LLaMA-Factory instruction to get the vanilla model inference result on `cross.tc.{chunksize}.{top_k}.json`:
```bash
llamafactory-cli train \
    --model_name_or_path {model_name} \
    --template {model_template} \
    --flash_attn auto \
    --dataset_dir ./data/test/crossner-tc/{model_name} \
    --dataset cross.tc.{chunksize}.{top_k} \
    --cutoff_len 8192 \
    --max_samples 1000000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 1024 \
    --top_p 0.1 \
    --temperature 0.01 \
    --repetition_penalty 1.1 \
    --output_dir ./data/test/crossner-results/{model_name} \
    --do_predict True \
    --fp16 True
```
7. Rename the inference result to `cross.tc.{chunksize}.{top_k}.jsonl` and put it in `./data/test/crossner-results/{model_name}`

# Evaluate
1. Run the CLI and enter  `evaluate` then choose `Evaluate the results on crossner.`