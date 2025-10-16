python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/big_run_32b_TEST \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/big_runs/final_images.json \
  --comparison_prompt_question '"Which image do you prefer looking at?"'

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/big_run_72b \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/big_runs/final_images.json \
  --comparison_prompt_question '"Which image do you prefer looking at?"'
