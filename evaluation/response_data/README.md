# Response Data

We collected the response data of AlpaGasus2-QLoRA for evaluating and comparing the responses with Alpaca2.
Please run the 'generate.py' following the instructions below, if you wanted to generate the responses of AlpaGasus2-QLoRA.

```
python generate.py \
    --base_model 'meta-llama/Llama-2-13b-hf' \
    --lora_weight 'StudentLLM/Alpagasus-2-13B-QLoRA' \
    --auth_token 'your authorization token' \
    --file_path 'AlpaGasus2-QLoRA/test_data/' \
    --save_path 'your data saving path' \    # You should specify the concrete path of saving point and jsonl file.
```
