#Data Preprocessing(convert test_file to SWAG pattern)
python preprocessing.py --context_file "${1}"  --test_file "${2}" 

#MC_prediction

python mc_test.py \
  --model_name_or_path mc_best_model \
  --do_predict \
  --max_seq_length 512 \
  --test_file mc_test.json \
  --output_file mc_prediction.json \
  --output_dir mc_test
  
#QA_prediction  

python qa_test.py \
  --model_name_or_path qa_best_model \
  --do_predict \
  --max_seq_length 512 \
  --test_file mc_prediction.json \
  --per_device_eval_batch_size 32 \
  --output_dir qa_predict \
  --output_file "${3}"

