python run_experiments.py --experiments compute_utilities --models qwen25-vl-7b-instruct --multimodal --additional_args --options_path ../../shared_options/multimodal_options_hierarchical.json


python run_experiments.py --experiments compute_utilities --models qwen25-vl-72b-instruct --multimodal --additional_args --options_path ../../shared_options/multimodal_options_hierarchical.json





python run_experiments.py --experiments compute_utilities --models qwen25-vl-7b-instruct --multimodal --additional_args --options_path ../../shared_options/multimodal_options_hierarchical.json

python run_experiments.py --experiments compute_utilities --models qwen25-vl-7b-instruct --multimodal --additional_args --options_path ../../shared_options/wikiart_style.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/ai_gen_img \
  --additional_args --options_path ../../shared_options/ai_gen_img.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/wikiart_style \
  --additional_args --options_path ../../shared_options/wikiart_style.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/combined_set \
  --additional_args --options_path ../../shared_options/combined_set.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-7b-instruct \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/t_7b \
  --additional_args --options_path ../../shared_options/options_hierarchical.json



export CUDA_VISIBLE_DEVICES=0
python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-7b-instruct \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/db_cb_7b \
  --additional_args --options_path ../../shared_options/combined_set.json


export CUDA_VISIBLE_DEVICES=1,2,3,4
python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/db_cb_72b \
  --additional_args --options_path ../../shared_options/combined_set.json



python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/db_wiki_72b \
  --additional_args --options_path ../../shared_options/wikiart_style.json




python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --override_gpu_count 4 \
  --time_limit 3-00:00:00 \
  --partition cais \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/wiki_style_72b \
  --overwrite_results \
  --additional_args --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/wikiart_style.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --override_gpu_count 8 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --multimodal \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/wiki_style_72b \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_k5 \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/wikiart_style.json



python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-7b-instruct \
  --slurm \
  --override_gpu_count 1 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/text_7b \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/options_hierarchical.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-7b-instruct \
  --multimodal \
  --overwrite_results \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/mm_7b \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/imagenet_test.json
                   



python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 4 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_race \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/fair_face_race.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 4 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_gender \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/fair_face_gender.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 4 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_age \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/fair_face_age.json

python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 4 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_wiki_all \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/wikiart_style.json


python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 4 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/cb_oi \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/cb_oi.json




python run_experiments.py \
  --experiments compute_utilities \
  --models qwen25-vl-72b-instruct \
  --slurm \
  --multimodal \
  --override_gpu_count 4 \
  --time_limit 1-00:00:00 \
  --partition cais \
  --output_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/ai_gen_img_256 \
  --overwrite_results \
  --additional_args --compute_utilities_config_key thurstonian_active_learning_save_ckpt \
                   --options_path /data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/ai_gen_img_256.json


