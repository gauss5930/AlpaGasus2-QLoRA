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
AlpaGauss2-QLoRA evaluation was processed with Eleuther AI's [LLM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)!
The evaluation tasks are tasks that are specified in HF's Open LLM Leaderboard.


