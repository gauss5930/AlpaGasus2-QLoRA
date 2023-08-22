for eval_file in 'koala_seed_0.json'
do
    python eval.py -qa ./response_data/results/${eval_file} -k1 alpaca2 -k2 alpagasus2 --batch_size 10 --max_tokens 256 --output_dir ./rating_data/
done
