python search_bias.py \
    --num_gender_words 200 \
    --num_wiki_words 5000 \
    --model_name_or_path bert-base-uncased \
    --output_dir ./out/ \
    --run_name run00 \
    --seed 42 \
    --per_device_batch_size 2048