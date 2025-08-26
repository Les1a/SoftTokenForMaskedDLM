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

# parallel dual branch
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args main_threshold=0.9,spec_threshold=0.6,merge_window=3,evolution_interval=4,model_path=${model_path},save_dir="./eval_results/parallel_prefix_dual_branch/${task}_0.6",gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True \
--output_path ./eval_results/parallel_prefix_dual_branch/${task}_0.6 \

# parallel dual branch
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args main_threshold=0.9,spec_threshold=0.9,merge_window=3,evolution_interval=4,model_path=${model_path},save_dir="./eval_results/parallel_prefix_dual_branch/${task}_0.8",gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,dual_branch=True,use_cache=True \
--output_path ./eval_results/parallel_prefix_dual_branch/${task}_0.8 \

# parallel
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},save_dir="./eval_results/parallel_prefix/${task}",gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True,use_cache=True \
--output_path ./eval_results/parallel_prefix/${task} \

