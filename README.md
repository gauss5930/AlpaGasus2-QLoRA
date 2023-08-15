# AlpaGasus2-QLoRA 🦙🦄🤏
This is an unofficial implementation of 'AlpaGasus: Training a better Alpaca with Fewer Data.' with LLaMA2 & QLoRA! The trained model is available at the [HuggingFace Hub](https://huggingface.co/StudentLLM)

This repository contains the source codes implementing AlpaGasus2-QLoRA with LLaMA2 and QLoRA.
Model size variants are 7B and 13B, for each of them we used [LLaMA2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [LLaMA2-13B-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf). 
For the dataset, [gpt4life](https://github.com/gpt4life/alpagasus)'s alpaca_t45 dataset filtered by gpt-3.5-turbo-0301 was utilized.

For implementing AlpaGasus2-QLoRA, Google Colab's single A100 40G GPU was used! 
In addition, we used [QLoRA](https://arxiv.org/abs/2305.14314) to implement the large model in only one GPU.
For implementing AlpaGasus2-QLoRA with QLoRA, HuggingFace's PEFT and BitsAndBytes library were utilized.
Further, the SFTTrainer of trl library was used to fine-tune the model.

## Dataset
AlpaGasus carefully selected higher-quality data through filtering on the original Alpaca instruction dataset to show improved performance than the original Alpaca.
For data filtering, gpt-3.5-turbo was used, as a result, the dataset was filtered from 52K to 9K.
Please check more specific data filtering processes in the [AlpaGasus paper](https://arxiv.org/abs/2307.08701), and [gpt4life](https://github.com/gpt4life/alpagasus)'s gpt-3.5-turbo filtered dataset, 'alpaca_t45.json' was used for fine-tuning AlpaGasus2-QLoRA.
Configuration of the dataset is as follows:

```
{
    'instruction': Give the instruction describing the question.
    'input': Occasionally present, detailed instructions accompany the question if available.
    'output': Give answers to questions.
}
.
.
.
```

## Requirements
If you want to run finetune.py, you need to install some libraries specified in 'requirements.txt'.

```
pip install -r requirements.txt
```

## Fine-tuning
We fine-tuned our model using the standard Hugging Face training code and referred to [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
AlpaGasus2-QLoRA was fine-tuned with LLaMA2-7B and LLaMA2-13B with following parameters:

|Hyperparameters|LLaMA2-7B|LLaMA2-13B|
|---|---|---|
|Batch size|128|128|
|learning rate|2e-5|1e-5|
|Epochs|3|5|
|Max Length|256|256|
|weight decay|0|0|

In addition, we also used QLoRA to save memory and speed up the fine-tuning of LLMs.
QLoRA Configuration is as follows:

|Hyperparameters|QLoRA|
|---|---|
|Quantization bit|4bit|
|Quantization type|NF4|
|LoRA rank|8|
|LoRA alpha|32|
|LoRA dropout|0.05|
|target modules|q_proj, v_proj|


- For the instruction-finetuning of LLaMA-2-7B:
```
python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path 'AlpaGasus2-QLoRA/dataset/alpaca_t45.json' \
    --data_type 'json' \
    --output_dir './results' \
    --hub_path 'Hub path to upload the model'
    --auth_token 'your HuggingFace Authorization code' \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --val_set_size 0.05
```

- For the instruction-finetuning of LLaMA-2-13B:
```
python finetune.py \
    --base_model 'meta-llama/Llama-2-13b-hf' \
    --data_path 'AlpaGasus2-QLoRA/dataset/alpaca_t45.json' \
    --data_type 'json' \
    --output_dir './results' \
    --hub_path 'Hub path to upload the model'
    --auth_token 'your HuggingFace authorization key'
    --num_epochs 5 \
    --learning_rate 1e-5
    --val_set_size 0.05
```

You can modify the arguments according to your taste!
```
python finetune.py \
    --base_model 'your model' \
    --data_path 'your data' \
    --data_type 'your data's type' \
    --hub_path 'Hub path to upload the model'
    --auth_token 'your HuggingFace authorization key'
    --output_dir './results' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --cutoff_len 512 \
    --val_set_size 0.05 \
    --load_in_4bit True \
    --bnb_4bit_quant_type 'nf4' \
    --bnb_4bit_double_quant True \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
```

## Response Data
Please check more specific information [here](https://github.com/gauss5930/AlpaGasus2-QLoRA/tree/main/evaluation/AlpaGasus%20Evaluation/response_data)!

## AlpaGasus2-QLoRA Evaluation
### 1. AlpaGasus Evaluation
We proceeded with the AlpaGasus Evaluation provided by [gpt4life](https://github.com/gpt4life/alpagasus/tree/main).
ChatGPT was used to grade the response of AlpaGasus2-QLoRA.

```
export OPEN_AI_KEY
cd evaluation/AlpaGasus Evaluation
sh run_eval.sh
```

### 2. Open LLM Leaderboard Evaluation
AlpaGauss2-QLoRA performance was uploaded on HuggingFace's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 
The evaluation task used the tasks specified in HF's Open LLM Leaderboard. (ARC, HellaSwag, MMLU, TruthfulQA)
The table shows the performance of AlpaGasus2-QLoRA on several benchmarks.

**Coming Soon!**

|Benchmarks|7B|13B|
|---|---|---|
|ARC|||
|HellaSwag|||
|MMLU|||
|TruthfulQA|||

## References
- [Llama2](https://arxiv.org/abs/2307.09288)
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/tree/main)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [AlpaGasus](https://arxiv.org/abs/2307.08701)
- [gpt4life/alpagasus](https://github.com/gpt4life/alpagasus)

## Citation
If you find it is a useful repository, please cite the paper:
```
@article{chen2023alpagasus,
  title={AlpaGasus: Training a Better Alpaca with Fewer Data},
  author={Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, Hongxia Jin},
  journal={arXiv preprint arXiv:2307.08701},
  year={2023}
}
```
