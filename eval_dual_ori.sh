# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=mbpp
length=256
block_length=32
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'

spec_threshold_list=(0.70 0.90 0.75 0.80 0.85 0.60 0.50)

# --- Loop through each threshold value ---
for threshold in "${spec_threshold_list[@]}"
do
  echo "======================================================"
  echo "Running evaluation: ${threshold} threshold"
  echo "======================================================"
  
  output_dir_vanilla="./eval_results/parallel_dual/${task}_${threshold}"

  accelerate launch eval_llada.py --tasks ${task} \
  --model llada_dist \
  --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir_vanilla}" \
  --output_path "${output_dir_vanilla}"
  # target_file=$(find "${output_dir_vanilla}/GSAI-ML__LLaDA-8B-Instruct" -type f -name "samples_humaneval_*.jsonl" | head -n 1)
  # python postprocess_code.py ${target_file}
done