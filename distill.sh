# !/bin/bash

# You need to specify these configurations
teacher_path='PATH_TO_TEACHER_MODEL'
general_student_path='PATH_TO_GENERAL_STUDENT_MODEL'
train_features_file='FEATURES_FILE_OF_TRAIN_EXAMPLES'
output_student_path='PATH_TO_OUTPUT_STUDENT_MODEL'
cache_path='./cache'
distill_model_type='simplified'

# Run distilling
python3 task_distill_simplified.py \
  --device '0' \
  --features_file ${train_features_file} \
  --teacher_model ${teacher_path} \
  --general_student_model ${general_student_path} \
  --output_student_dir ${output_student_path} \
  --cache_file_dir ${cache_path} \
  --max_seq_length 256 \
  --do_lower_case \
  --train_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --train_loss_step 1000 \
  --save_model_step 3000 \
#  --fp16
# If you set '--fp16', install 'apex' first.