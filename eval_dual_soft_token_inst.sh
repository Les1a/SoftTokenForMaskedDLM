# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# conda activate ...
# cd /home/...

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cuda_visible <<< "${CUDA_VISIBLE_DEVICES// /}"
  available_gpu_num=${#_cuda_visible[@]}
elif command -v nvidia-smi >/dev/null 2>&1; then
  available_gpu_num=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
else
  available_gpu_num=0
fi

task_list=(mbpp gsm8k)
length=256
block_length=32
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

threshold_list=(0.70 0.90 0.75 0.80 0.85)

# --- soft token evaluation ---
for task in "${task_list[@]}"
do
  for threshold in "${threshold_list[@]}"
  do
    echo "======================================================"
    echo "Running evaluation: ${task} dataset, ${threshold} threshold"
    echo "======================================================"
    
    output_dir="./eval_results_soft_token/inst_parallel_dual/${task}_${threshold}"

    accelerate launch --num_processes=${available_gpu_num} eval_llada.py --tasks ${task} \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --model_args model_path=${model_path},soft_token=True,gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
    --output_path "${output_dir}"
  done
done

# --- baseline evaluation ---
for task in "${task_list[@]}"
do
  for threshold in "${threshold_list[@]}"
  do
    echo "======================================================"
    echo "Running evaluation: ${task} dataset, ${threshold} threshold"
    echo "======================================================"
    
    output_dir="./eval_results_baseline/inst_parallel_dual/${task}_${threshold}"

    accelerate launch --num_processes=${available_gpu_num} eval_llada.py --tasks ${task} \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
    --output_path "${output_dir}"
  done
done