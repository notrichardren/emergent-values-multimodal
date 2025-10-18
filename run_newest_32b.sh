

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/company_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/company_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/country_flags \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt_small \
  --options_path /data/superstimuli_group/richard/small_runs/country_flags.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/food101_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/food101_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/imagenet_o_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/imagenet_o_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/logos_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/logos_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/politicians_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/politicians_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/politicians_large_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/politicians_large_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/resolution_test_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt_small \
  --options_path /data/superstimuli_group/richard/small_runs/resolution_test_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/species_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/species_final.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/wikiart_final \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/small_runs/wikiart_final.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-32b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/richard_ren/emergent-values-multimodal/outputs/big_run_2 \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
  --options_path /data/superstimuli_group/richard/big_run_2/all_images_final.json