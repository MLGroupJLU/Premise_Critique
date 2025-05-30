export HF_ENDPOINT=https://hf-mirror.com
# 需要在主目录运行 bash -x evaluation/eval_pipeline.sh

# ollama 均非流式才能正常统计usage

# Qwen/Qwen3-30B-A3B(可以调用api，qwen3-30b-a3b) -----------------------------------------------(流式输出有误，Qwen官方可以输出)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-30b-a3b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --temperature 0.7 --top_p 0.8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-30b-a3b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# eval-inference
python ./evaluation/eval.py --model_name qwen3-30b-a3b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-30b-a3b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-235B-A22B(可以调用api，qwen3-235b-a22b) -----------------------------------------------(流式输出有误，Qwen官方可以输出)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-235b-a22b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream  --temperature 0.7 --top_p 0.8
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-235b-a22b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --temperature 0.7 --top_p 0.8
# eval-inference
python ./evaluation/eval.py --model_name qwen3-235b-a22b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-235b-a22b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream
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

# Qwen/Qwen3-235B-A22B(可以调用api，qwen3-30b-a3b) -----------------------------------------------(流式输出有错，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-235b-a22b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-235b-a22b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-235b-a22b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-235b-a22b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
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

# Qwen/Qwen3-235B-A22B(可以调用api，qwen3-30b-a3b) -----------------------------------------------(流式输出有错，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-235b-a22b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-235b-a22b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-235b-a22b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-235b-a22b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------

# Qwen/Qwen3-8B-thinking(可以调用api，qwen3-8b) -----------------------------------------------(流式输出有错，Qwen官方可以输出，但无think_count，输出爆炸)
# infer-inference
python ./evaluation/inference.py --model_name qwen3-8b --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --enable_thinking --temperature 0.6 --top_p 0.95
# infer-check 直到没有样本被遗漏
python ./evaluation/inference.py --model_name qwen3-8b --mode check --save_frequency 1 --dataset_load_proc 10 --infer_proc 20 --stream --enable_thinking --temperature 0.6 --top_p 0.95
# eval-inference
python ./evaluation/eval.py --model_name qwen3-8b --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10 --stream --enable_thinking
# eval-check 直到没有样本被遗漏
python ./evaluation/eval.py --model_name qwen3-8b --mode check --evaluator gpt-4.1 --save_frequency 2 --infer_proc 1 --stream --enable_thinking
# statistics
python ./evaluation/statistics.py
# ----------------------------------------------------------