for eval_file in 'koala_seed_0.json', 'sinstruct_seed_0.json', 'vicuna_seed_0.json'
do
    python evaluation.py -qa AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/response_data/results/${eval_file} -k1 alpaca2 -k2 alpagasus2 --batch_size 10 --max_tokens 256 --output_dir AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/rating_data/
    python evaluation.py -qa AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/response_data/results/${eval_file} -k1 alpagasus2 -k2 alpaca2 --batch_size 10 --max_tokens 256 --output_dir AlpaGasus2-QLoRA/evaluation/AlpaGasus-Evaluation/rating_data/
done
