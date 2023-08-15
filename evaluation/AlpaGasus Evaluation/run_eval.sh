for eval_file in 'koala_seed_0.json', 'sinstruct_seed_0.json', 'vicuna_seed_0.json'
do
    python eval.py -qa ./results/${eval_file} -k1 alpaca2 -k2 alpagasus2 --batch_size 10 --max_tokens 256 --output_dir ./rating_data/
    python eval.py -qa ./results/${eval_file} -k1 alpagasus2 -k2 alpaca2 --batch_size 10 --max_tokens 256 --output_dir ./rating_data/
done
