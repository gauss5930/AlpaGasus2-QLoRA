# AlpaGasus Evaluation

We tried to follow the evaluation metric introduced by AlpaGasus paper.
During the process, we consulted to the code by [gpt4life](https://github.com/gpt4life/alpagasus/blob/main/evaluation/eval.py), unofficial implementation of AlpaGasus.
Due to resource constraints, we have replaced the evaluator model from GPT-4 to gpt-3.5-turbo.

### Running evaluation.py

The provided code below is for the running of evaluation.py. When executing the code, please ensure to input your own OpenAI API key.

```python
%cd AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation
for eval_file in 'koala_seed_0.json', 'sinstruct_seed_0.json', 'vicuna_seed_0.json':
    python evaluation.py --API_KEY your_own_API_KEY -qa AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/response_data/results/${eval_file} -k1 alpaca2 -k2 alpagasus2 --max_tokens 256 --output_dir AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/rating_data/
    python evaluation.py --API_KEY your_own_API_KEY -qa AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/response_data/results/${eval_file} -k1 alpagasus2 -k2 alpaca2 --max_tokens 256 --output_dir AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/rating_data/ 
```
