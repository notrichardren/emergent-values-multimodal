
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python run_experiments.py --experiments compute_utilities --models qwen25-vl-72b-instruct --multimodal --additional_args --options_path ../../shared_options/wikiart_style.json


