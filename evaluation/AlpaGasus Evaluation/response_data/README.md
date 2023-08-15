# Response Data

We collected the response data of AlpaGasus2-QLoRA for evaluating and comparing the responses with Alpaca2.
Please run the 'generate.py' following the instructions below to collect the response data.

```python
for model_type in 'alpaca2', 'alapgasus2'
    do
        python generate.py \
            --base_model 'meta-llama/Llama-2-13b-hf' \
            --lora_weight model_type \
            --auth_token 'your authorization token' \
            --file_path 'AlpaGasus2-QLoRA/test_data/' \
            --save_path './results/' \    
    done
```
