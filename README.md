# AlpaGasus2-QLoRA 🦙🦄🤏
This is an unofficial implementation of 'AlpaGasus: Training a better Alpaca with Fewer Data.' with LLaMA2 & QLoRA! The trained model is available at the [HuggingFace Hub]()

This repository contains the source codes that implement AlpaGasus2-QLoRA using LLaMA2 and QLoRA.
Model size variants are 7B and 13B, for each of them we used LLaMA2-7B and LLaMA2-13B. 
For the dataset, [gpt4life](https://github.com/gpt4life/alpagasus)'s alpaca_t45 dataset filtered by fpt-3.5-turbo-0301 was utilized to implement AlpaGasus2-QLoRA.

For implementing AlpaGasus2-QLoRA, Google Colab's single A100 80G GPU was used. 
In addition, we used QLoRA to implement the large model in only one GPU.
For implementing AlpaGasus2-QLoRA with QLoRA, HuggingFace's PEFT and BitsAndBytes library were utilized.

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

## Fine-tuning
We fine-tuned our model using the standard Hugging Face training code and referred to [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
AlpaGasus2-QLoRA was fine-tuned with LLaMA2-7B and LLaMA2-13B with following parameters:

|Hyperparameters|LLaMA2-7B|LLaMA2-13B|
|---|---|---|
|Batch size|128|128|
|learning rate|2e-5|1e-5|
|Epochs|3|5|
|Max Length|512|512|
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

## AlpaGasus2-QLoRA Evaluation
### 1. Open LLM Leaderboard Evaluation
AlpaGauss2-QLoRA performance was uploaded on HuggingFace's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 
The evaluation task used the tasks specified in HF's Open LLM Leaderboard. (ARC, HellaSwag, MMLU, TruthfulQA)
The table shows the performance of AlpaGasus2-QLoRA on several benchmarks.

|Benchmarks|7B|13B|
|---|---|---|
|ARC|||
|HellaSwag|||
|MMLU|||
|TruthfulQA|||

## References
- [Llama2](https://arxiv.org/abs/2307.09288)
- [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [AlpaGasus](https://arxiv.org/abs/2307.08701)
- [MT-Bench](https://arxiv.org/abs/2306.05685)
- [gpt4life/alpagasus](https://github.com/gpt4life/alpagasus)

## Citation
If you find our repository useful, please cite the paper:
```
@article{chen2023alpagasus,
  title={AlpaGasus: Training a Better Alpaca with Fewer Data},
  author={Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, Hongxia Jin},
  journal={arXiv preprint arXiv:2307.08701},
  year={2023}
}
```

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
