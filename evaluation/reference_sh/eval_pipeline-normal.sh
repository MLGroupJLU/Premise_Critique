export HF_ENDPOINT=https://hf-mirror.com
# 需要在主目录运行 bash -x evaluation/eval_pipeline.sh

# 普通模型

# gpt-4o -----------------------------------------------(正常输出，非流式，完成推理)
# infer-inference
python ./evaluation/inference.py --model_name gpt-4o --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gpt-4o --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1 
# eval-inference
python ./evaluation/eval.py --model_name gpt-4o --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gpt-4o --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# gpt-4.1 -----------------------------------------------(正常输出，非流式，完成推理)
# infer-inference
python ./evaluation/inference.py --model_name gpt-4.1 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gpt-4.1 --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 5 
# eval-inference
python ./evaluation/eval.py --model_name gpt-4.1 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gpt-4.1 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# gpt-4.1-mini -----------------------------------------------(正常输出，非流式，完成推理，需要check)
# infer-inference
python ./evaluation/inference.py --model_name gpt-4.1-mini --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gpt-4.1-mini --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# eval-inference
python ./evaluation/eval.py --model_name gpt-4.1-mini --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gpt-4.1-mini --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-8B(可以调用api，qwen3-8b) -----------------------------------------------(流式输出有错,Qwen官方可以输出) (silconcloud)
# infer-inference
python ./evaluation/inference.py --model_name Qwen/Qwen3-8B --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 5 --stream --temperature 0.7 --top_p 0.8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name Qwen/Qwen3-8B --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# eval-inference
python ./evaluation/eval.py --model_name Qwen3-8B --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name Qwen3-8B --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-14B(可以调用api，qwen3-14b) -----------------------------------------------(正常输出，流式，Qwen官方可以输出)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-14b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-14b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# eval-inference
python ./evaluation/eval.py --model_name qwen3-14b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-14b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-32B(可以调用api，qwen3-32b) -----------------------------------------------(正常输出，流式，Qwen官方可以输出)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-32b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# infer-check 直到没有样本被遗漏 
python ./evaluation/inference.py --model_name qwen3-32b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# eval-inference
python ./evaluation/eval.py --model_name qwen3-32b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-32b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-30B-A3B(可以调用api，qwen3-30b-a3b) -----------------------------------------------(流式输出有误，Qwen官方可以输出)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-30b-a3b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-30b-a3b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# eval-inference
python ./evaluation/eval.py --model_name qwen3-30b-a3b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-30b-a3b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# DeepSeek-V3(可以调用api，deepseek-ai/DeepSeek-V3) -----------------------------------------------(正常输出，非流式，完成推理)
# infer-inference
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-V3 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-V3 --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# eval-inference
python ./evaluation/eval.py --model_name DeepSeek-V3 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name DeepSeek-V3 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# LLM-Research/Llama-4-Scout-17B-16E-Instruct(可以调api，meta-llama/llama-4-scout) -----------------------------------------------(正常输出，非流式，完成推理)
# infer-inference
python ./evaluation/inference.py --model_name meta-llama/llama-4-scout --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 20 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name meta-llama/llama-4-scout --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1 
# eval-inference
python ./evaluation/eval.py --model_name llama-4-scout --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name llama-4-scout --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# LLM-Research/Llama-4-Maverick-17B-128E-Instruct(可以调api，meta-llama/llama-4-maverick) -----------------------------------------------(正常输出，非流式，完成推理)
# infer-inference
python ./evaluation/inference.py --model_name meta-llama/llama-4-maverick --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name meta-llama/llama-4-maverick --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# eval-inference
python ./evaluation/eval.py --model_name llama-4-maverick --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name llama-4-maverick --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# LLM-Research/gemma-3-27b-it(可以调用api，gemma-3-27b-it) -----------------------------------------------(正常输出，非流式，部分完成推理)
# infer-inference
python ./evaluation/inference.py --model_name gemma-3-27b-it --mode inference --save_frequency 1 --dataset_load_proc 8 --infer_proc 8 
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gemma-3-27b-it --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1 
# eval-inference
python ./evaluation/eval.py --model_name gemma-3-27b-it --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gemma-3-27b-it --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# gemini-2.5-flash-preview-04-17-nothinking -----------------------------------------------(网络连接问题，部分完成推理)
# infer-inference
python ./evaluation/inference.py --model_name gemini-2.5-flash-preview-04-17-nothinking --mode inference --save_frequency 1 --dataset_load_proc 8 --infer_proc 8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gemini-2.5-flash-preview-04-17-nothinking --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1 
# eval-inference
python ./evaluation/eval.py --model_name gemini-2.5-flash-preview-04-17-nothinking --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gemini-2.5-flash-preview-04-17-nothinking --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# claude-3-7-sonnet-20250219 -----------------------------------------------(正常输出，非流式)
# infer-inference
python ./evaluation/inference.py --model_name claude-3-7-sonnet-20250219 --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name claude-3-7-sonnet-20250219 --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1 
# eval-inference
python ./evaluation/eval.py --model_name claude-3-7-sonnet-20250219 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name claude-3-7-sonnet-20250219 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# THUDM/GLM-4-9B-0414 -----------------------------------------------(silconcloud)
# infer-inference
python ./evaluation/inference.py --model_name THUDM/GLM-4-9B-0414 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 5
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name THUDM/GLM-4-9B-0414 --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# eval-inference
python ./evaluation/eval.py --model_name GLM-4-9B-0414 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name GLM-4-9B-0414 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# other models...
