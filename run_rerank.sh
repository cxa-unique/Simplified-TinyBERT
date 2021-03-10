# !/bin/bash

model_path='PATH_TO_MODEL'
features_file='FEATURES_FILE_FOR_TEST_PAIRS'
output_scores_file='OUTPUT_SCORES_FILE'

python3 run_rerank.py \
    --device '0' \
    --model_dir ${model_path} \
    --features_file ${features_file} \
    --output_scores_file ${output_scores_file} \
    --cache_file_dir './cache'