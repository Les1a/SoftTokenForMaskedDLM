# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'

spec_threshold_list=(0.4 0.5 0.6 0.7 0.75)

# --- Loop through each threshold value ---
for threshold in "${spec_threshold_list[@]}"
do
  echo "======================================================"
  echo "Running evaluation with threshold: ${threshold}"
  echo "======================================================"

  output_dir="./eval_results/parallel_dual_dual_branch_0.75base/${task}_${threshold}"
  output_dir_vanilla="./eval_results/parallel_dual/${task}_${threshold}"

  accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args main_threshold=0.75,spec_threshold=${threshold},merge_window=3,evolution_interval=4,model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
  --output_path "${output_dir}"

  # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  # --confirm_run_unsafe_code --model llada_dist \
  # --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir_vanilla}" \
  # --output_path "${output_dir_vanilla}" 
done


spec_threshold_list=(0.5 0.6 0.7)

# --- Loop through each threshold value ---
for threshold in "${spec_threshold_list[@]}"
do
  echo "======================================================"
  echo "Running evaluation with threshold: ${threshold}"
  echo "======================================================"

  output_dir="./eval_results/parallel_dual_dual_branch_0.7base/${task}_${threshold}"
  output_dir_vanilla="./eval_results/parallel_dual/${task}_${threshold}"

  accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args main_threshold=0.7,spec_threshold=${threshold},merge_window=3,evolution_interval=4,model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
  --output_path "${output_dir}"

  # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  # --confirm_run_unsafe_code --model llada_dist \
  # --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir_vanilla}" \
  # --output_path "${output_dir_vanilla}" 
done


spec_threshold_list=(0.5 0.6 0.7 0.75 0.8)

# --- Loop through each threshold value ---
for threshold in "${spec_threshold_list[@]}"
do
  echo "======================================================"
  echo "Running evaluation with threshold: ${threshold}"
  echo "======================================================"

  output_dir="./eval_results/parallel_dual_dual_branch_0.8base/${task}_${threshold}"
  output_dir_vanilla="./eval_results/parallel_dual/${task}_${threshold}"

  accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args main_threshold=0.8,spec_threshold=${threshold},merge_window=3,evolution_interval=4,model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
  --output_path "${output_dir}"

  # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  # --confirm_run_unsafe_code --model llada_dist \
  # --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir_vanilla}" \
  # --output_path "${output_dir_vanilla}" 
done



spec_threshold_list=(0.5 0.6 0.7 0.75 0.8 0.85)

# --- Loop through each threshold value ---
for threshold in "${spec_threshold_list[@]}"
do
  echo "======================================================"
  echo "Running evaluation with threshold: ${threshold}"
  echo "======================================================"

  output_dir="./eval_results/parallel_dual_dual_branch_0.85base/${task}_${threshold}"
  output_dir_vanilla="./eval_results/parallel_dual/${task}_${threshold}"

  accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args main_threshold=0.85,spec_threshold=${threshold},merge_window=3,evolution_interval=4,model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
  --output_path "${output_dir}"

  # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  # --confirm_run_unsafe_code --model llada_dist \
  # --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir_vanilla}" \
  # --output_path "${output_dir_vanilla}" 
done



spec_threshold_list=(0.5 0.6 0.7 0.75 0.8 0.85 0.9)

# --- Loop through each threshold value ---
for threshold in "${spec_threshold_list[@]}"
do
  echo "======================================================"
  echo "Running evaluation with threshold: ${threshold}"
  echo "======================================================"

  output_dir="./eval_results/parallel_dual_dual_branch_0.9base/${task}_${threshold}"
  output_dir_vanilla="./eval_results/parallel_dual/${task}_${threshold}"

  accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args main_threshold=0.9,spec_threshold=${threshold},merge_window=3,evolution_interval=4,model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True,dual_cache=True,save_dir="${output_dir}" \
  --output_path "${output_dir}"

  # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  # --confirm_run_unsafe_code --model llada_dist \
  # --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},show_speed=True,use_cache=True,dual_cache=True,save_dir="${output_dir_vanilla}" \
  # --output_path "${output_dir_vanilla}" 
done