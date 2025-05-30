export HF_ENDPOINT=https://hf-mirror.com
# 需要在主目录运行 bash -x evaluation/eval_pipeline.sh

# 推理模型

# Qwen/QwQ-32B(可以调用api，Qwen/QwQ-32B-Preview) -----------------------------------------------(流式输出有错，非流式可以输出，无think_count)
# infer-inference
python ./evaluation/inference.py --model_name Qwen/QwQ-32B --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5 --stream
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name Qwen/QwQ-32B-Preview --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1 --stream
# eval-inference
python ./evaluation/eval.py --model_name Qwen/QwQ-32B-Preview --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name Qwen/QwQ-32B-Preview --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# o3-mini-medium -----------------------------------------------(可以正常输出，非流式，完成推理，需要check，存在异常换行符)
# infer-inference
python ./evaluation/inference.py --model_name o3-mini-medium --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name o3-mini-medium --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name o3-mini-medium --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name o3-mini-medium --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# o4-mini-medium -----------------------------------------------(可以正常输出，非流式)
# infer-inference
python ./evaluation/inference.py --model_name o4-mini-medium --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name o4-mini-medium --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name o4-mini-medium --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name o4-mini-medium --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Deepseek-R1(可以调用api，deepseek-ai/DeepSeek-R1) -----------------------------------------------(可以正常输出，非流式，思维链长度存在上限)
# infer-inference
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 1
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1 --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name DeepSeek-R1 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name DeepSeek-R1 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-8B-thinking(可以调用api，qwen3-8b) -----------------------------------------------(流式输出有错，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name Qwen/Qwen3-8B --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-8b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-8b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-8b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-14B-thinking(可以调用api，qwen3-14b) -----------------------------------------------(可以正常输出，流式，无think_count，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-14b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-14b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-14b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-14b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-32B-thinking(可以调用api，qwen3-32b) -----------------------------------------------(可以正常输出，流式，无think_count，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-32b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-32b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-32b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-32b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-30B-A3B-thinking(可以调用api，qwen3-30b-a3b) -----------------------------------------------(流式输出有错，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-30b-a3b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-30b-a3b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-30b-a3b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-30b-a3b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# claude-3-7-sonnet-20250219-thinking -----------------------------------------------(没有thinking_count，两种输出正常)
# infer-inference
python ./evaluation/inference.py --model_name claude-3-7-sonnet-20250219-thinking --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name claude-3-7-sonnet-20250219-thinking --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name claude-3-7-sonnet-20250219-thinking --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name claude-3-7-sonnet-20250219-thinking --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# claude-3-7-sonnet-latest -----------------------------------------------(没有thinking_count，两种输出正常)
# infer-inference
python ./evaluation/inference.py --model_name claude-3-7-sonnet-latest --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name claude-3-7-sonnet-latest --mode check --save_frequency 2 --dataset_load_proc 10 --infer_proc 1
# eval-inference
python ./evaluation/eval.py --model_name claude-3-7-sonnet-latest --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name claude-3-7-sonnet-latest --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# gemini-2.5-flash-preview-04-17-thinking -----------------------------------------------(网络连接问题)
# infer-inference
python ./evaluation/inference.py --model_name gemini-2.5-flash-preview-04-17-thinking --mode inference --save_frequency 1 --dataset_load_proc 8 --infer_proc 8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gemini-2.5-flash-preview-04-17-thinking --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name gemini-2.5-flash-preview-04-17-thinking --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gemini-2.5-flash-preview-04-17-thinking --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# gemini-2.5-pro-exp-03-25 -----------------------------------------------(流式输出错误；非流式正常输出，没有think_count)
# infer-inference
python ./evaluation/inference.py --model_name gemini-2.5-pro-exp-03-25 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name gemini-2.5-pro-exp-03-25 --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name gemini-2.5-pro-exp-03-25 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name gemini-2.5-pro-exp-03-25 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# DeepSeek-R1-Distill-Llama-70B(可以调用api，deepseek-ai/DeepSeek-R1-Distill-Llama-70B) -----------------------------------------------(可以正常输出结果，非流式，速度很慢)
# infer-inference
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-70B --mode inference --save_frequency 1 --dataset_load_proc 8 --infer_proc 8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-70B --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name DeepSeek-R1-Distill-Llama-70B --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name DeepSeek-R1-Distill-Llama-70B --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# phi-4-reasoning-plus(可以调用api，phi-4-reasoning-plus，存在格式要求) -----------------------------------------------（网络连接有误）
# infer-inference
python ./evaluation/inference.py --model_name phi-4-reasoning-plus --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name phi-4-reasoning-plus --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20
# eval-inference
python ./evaluation/eval.py --model_name phi-4-reasoning-plus --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name phi-4-reasoning-plus --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# THUDM/GLM-Z1-9B-0414 -----------------------------------------------(silconcloud)
# infer-inference
python ./evaluation/inference.py --model_name THUDM/GLM-Z1-9B-0414 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 5
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name THUDM/GLM-Z1-9B-0414 --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# eval-inference
python ./evaluation/eval.py --model_name GLM-Z1-9B-0414 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name GLM-Z1-9B-0414 --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -----------------------------------------------(silconcloud)
# infer-inference
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 5
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 
# eval-inference
python ./evaluation/eval.py --model_name DeepSeek-R1-Distill-Qwen-7B --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name DeepSeek-R1-Distill-Qwen-7B --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------


# other models...