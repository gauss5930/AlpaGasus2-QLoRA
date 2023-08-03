# AlpaGasus2-QLoRA 🦙🦄🤏
This is an unofficial implementation of 'AlpaGasus: Training a better Alpaca with Fewer Data.' with LLaMA2 & QLoRA! The trained model is available at the [HuggingFace Hub]()

This repository contains the source codes that implement AlpaGasus2-QLoRA using LLaMA2 and QLoRA.
Model size variants are 7B and 13B, and [gpt4life](https://github.com/gpt4life/alpagasus)'s alpaca_filtered_data used to implementation of AlpaGasus was utilized.
The composition of the alpaca_filtered_dataset has a few differences compared to the original filtered dataset of AlpaGasus. 
This is because the alpaca_filtered_dataset was filtered by Claude however, the original filtered dataset was filtered by gpt-3.5-turbo.

For implementing AlpaGasus2-QLoRA, Google Colab's single A100 80G GPU was used. 
To implement the large model in only one GPU, HuggingFace's PEFT and BitsAndBytes library were utilized. 

## Dataset

## Fine-tuning

## Parameters

## AlpaGasus2-QLoRA Evaluation
### 1. Open LLM Leaderboard Evaluation
AlpaGauss2-QLoRA evaluation was uploaded on HuggingFace's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 
The evaluation tasks are tasks that are specified in HF's Open LLM Leaderboard. (ARC, HellaSwag, MMLU, TruthfulQA)
The table shows the performance of AlpaGasus2-QLoRA on several benchmarks.

## Reference
- [AlpaGasus](https://lichang-chen.github.io/AlpaGasus/)
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
