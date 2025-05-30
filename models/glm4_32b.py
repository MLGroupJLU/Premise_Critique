import os
import argparse
import re
import concurrent
from datasets import load_dataset
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from models.LLM import LLM


def load_data():
    """加载数据集并转换为字典（添加异常处理）"""
    try:
        dataset = load_dataset("ALIENS232/Premise-Critique", split="train", num_proc=10)
        return {data["pid"]: data for data in dataset}
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        return {}


def get_answer(llm, query, max_retries=10, retry_interval=5):
    """带重试机制的LLM响应获取（增加重试间隔和更明确的异常处理）"""
    for attempt in range(max_retries):
        try:
            response = llm.get_response([{"role": "user", "content": query}])
            return response.get("formal_answer", "") if response else ""
        except Exception as e:
            logging.warning(f"第{attempt+1}次API请求失败，{retry_interval}秒后重试: {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_interval)
    logging.error("所有API重试均失败")
    return ""


def get_conflict_place(meta_data):
    """安全的冲突位置获取（增加默认返回和类型检查）"""
    if not isinstance(meta_data, dict):
        return ""
    conflict_type = meta_data.get("conflict_type", "")
    conflict = meta_data.get("conflict", {})
    if conflict_type in ["contra_infer_insert", "contra_premise_insert"]:
        return conflict.get("conflict_reason", "")
    elif conflict_type in ["flawed_solution_completion", "irr_query_distraction"]:
        return f"Step '{conflict.get('recomposed_premise', '')}' in partial solution is wrong"
    return ""


def extract_json_data(eval_result):
    """安全的JSON提取（处理无匹配情况并返回默认结构）"""
    try:
        match = re.search(r'```json\s*([\s\S]*?)\s*```', eval_result, re.DOTALL)
        return json.loads(match.group(1)) if match else {
            "if_find_contradiction": "False",
            "basis": "No valid JSON response found"
        }
    except Exception as e:
        logging.error(f"JSON解析失败: {e}, 响应内容: {eval_result[:200]}")
        return {
            "if_find_contradiction": "False",
            "basis": "Invalid JSON format in response"
        }


def evaluate_answer_to_normal(llm, response, correct_answer=""):
    """评估普通回答（增加响应为空处理）"""
    if not response:
        return {"correctness": "False"}  # 无响应视为错误
    prompt = prompt_template_normal.format(response=response, correct_answer=correct_answer)
    eval_result = get_answer(llm, prompt)
    return {"correctness": "True" if "true" in str(eval_result).lower() else "False"}


def evaluate_answer_to_ill(llm, response, conflict_place):
    """评估ill类型回答（增加响应为空处理）"""
    if not response:
        return {
            "if_find_contradiction": "False",
            "basis": "No response provided for premise critique"
        }
    prompt = prompt_template_ill.format(conflict_place=conflict_place, response=response)
    eval_result = get_answer(llm, prompt)
    return extract_json_data(eval_result)


# 定义独立的提示模板常量
prompt_template_normal = """
Please act as an impartial judge...（保持原模板内容不变）
"""

prompt_template_ill = """
Please act as an impartial judge...（保持原模板内容不变）
"""


def evaluate_sample(llm, eval_data, infer_result, dataset_dict):
    """评估单个样本（增加状态检查避免重复评估）"""
    pid = eval_data["pid"]
    meta_data = dataset_dict.get(pid, {})
    conflict_place = get_conflict_place(meta_data)
    
    # 处理Normal评估（仅当未完成时执行）
    if not eval_data["GPT_eval_result"].get("normal") and dataset_dict[pid].get("conflict_type") != "irr_query_distraction":
        normal_response = infer_result.get("answer_to_normal", {}).get("formal_answer", "")
        eval_data["normal_answer_length"] = {
            "all_count": infer_result.get("answer_to_normal", {}).get("all_token_count", 0),
            "think_count": infer_result.get("answer_to_normal", {}).get("thinking_token_count", 0)
        }
        if normal_response:
            correct_answer = dataset_dict[pid].get("final_answer", "")
            eval_data["GPT_eval_result"]["normal"] = evaluate_answer_to_normal(llm, normal_response, correct_answer)
    
    # 处理Active评估（仅当未完成时执行）
    if not eval_data["GPT_eval_result"].get("active"):
        ill_response = infer_result.get("answer_to_ill", {}).get("formal_answer", "")
        eval_data["ill_answer_length"] = {
            "all_count": infer_result.get("answer_to_ill", {}).get("all_token_count", 0),
            "think_count": infer_result.get("answer_to_ill", {}).get("thinking_token_count", 0)
        }
        if ill_response:
            eval_data["GPT_eval_result"]["active"] = evaluate_answer_to_ill(llm, ill_response, conflict_place)
    
    # 处理Passive评估（仅当未完成时执行）
    if not eval_data["GPT_eval_result"].get("passive"):
        ill_with_hint_response = infer_result.get("answer_to_ill_with_hint", {}).get("formal_answer", "")
        eval_data["ill_with_hint_answer_length"] = {
            "all_count": infer_result.get("answer_to_ill_with_hint", {}).get("all_token_count", 0),
            "think_count": infer_result.get("answer_to_ill_with_hint", {}).get("thinking_token_count", 0)
        }
        if ill_with_hint_response:
            eval_data["GPT_eval_result"]["passive"] = evaluate_answer_to_ill(llm, ill_with_hint_response, conflict_place)
    
    return eval_data


def read_data_from_jsonl(file_path):
    """安全的JSONL读取（处理文件不存在和解析错误）"""
    if not os.path.exists(file_path):
        return []
    try:
        return [json.loads(line.strip()) for line in open(file_path, 'r', encoding='utf-8') if line.strip()]
    except Exception as e:
        logging.error(f"读取{file_path}失败: {e}")
        return []


def write_to_file(data, file_path):
    """安全的JSONL写入（处理文件写入错误）"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"成功写入{len(data)}条数据到{file_path}")
    except Exception as e:
        logging.error(f"写入{file_path}失败: {e}")


def process_evaluation(llm, args, dataset_dict):
    """处理评估过程（增强断点续传和状态管理）"""
    infer_path = os.path.join("evaluation", "infer_result", f"{args.model_name}_infer_result.jsonl")
    save_path = os.path.join("evaluation", "eval_result", f"{args.model_name}_eval_result.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    infer_results = read_data_from_jsonl(infer_path)
    existing_evals = read_data_from_jsonl(save_path)
    
    # 初始化评估数据（保留已有评估结果）
    if not existing_evals:
        existing_evals = [{
            "pid": res["pid"],
            "GPT_eval_result": {k: {} for k in ["normal", "active", "passive"]},
            "normal_answer_length": {"all_count": 0, "think_count": 0},
            "ill_answer_length": {"all_count": 0, "think_count": 0},
            "ill_with_hint_answer_length": {"all_count": 0, "think_count": 0}
        } for res in infer_results]
    
    # 创建PID到推理结果的映射（处理可能的缺失）
    pid_to_infer = {res["pid"]: res for res in infer_results if "pid" in res}

    def worker(eval_data):
        pid = eval_data["pid"]
        infer_result = pid_to_infer.get(pid)
        if not infer_result:
            logging.warning(f"跳过缺失推理结果的样本: PID={pid}")
            return eval_data
        
        # 检查是否所有评估已完成
        all_done = all(v != {} for v in eval_data["GPT_eval_result"].values())
        if all_done:
            logging.info(f"样本{pid}评估已完成，跳过")
            return eval_data
        
        return evaluate_sample(llm, eval_data, infer_result, dataset_dict)

    # 多线程处理并支持中途保存
    results = existing_evals.copy()
    with ThreadPoolExecutor(max_workers=args.infer_proc) as executor:
        futures = [executor.submit(worker, eval_data) for eval_data in results]
        completed = 0
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评估进度"):
            try:
                updated_data = future.result()
                results[completed] = updated_data  # 保持顺序一致
                completed += 1
                
                # 按频率保存进度
                if completed % args.save_frequency == 0:
                    write_to_file(results[:completed], save_path)  # 仅保存已完成部分
                
            except Exception as e:
                logging.error(f"处理样本时发生错误: {e}")
        
        # 处理所有完成后保存完整结果
        write_to_file(results, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="o3-mini-medium")
    parser.add_argument("--evaluator", type=str, default="o3-mini-high")
    parser.add_argument("--save_frequency", type=int, default=100)  # 建议生产环境调大
    parser.add_argument("--infer_proc", type=int, default=4)
    parser.add_argument("--DEBUG", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    class ModelArgs:
        enable_thinking = False
        temperature = 0.0
        top_p = 1.0
        max_tokens = 32768
        n = 1
        stream = False

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][EVAL] %(message)s",
        filename="premise_critique.log",
        filemode="a",
        handlers=[logging.StreamHandler()]  # 同时输出到控制台
    )

    dataset_dict = load_data()
    if not dataset_dict:
        logging.error("数据集加载失败，程序终止")
        sys.exit(1)
    
    try:
        llm = LLM(model_name=args.evaluator, model_args=ModelArgs())
    except Exception as e:
        logging.error(f"LLM初始化失败: {e}")
        sys.exit(1)

    process_evaluation(llm, args, dataset_dict)