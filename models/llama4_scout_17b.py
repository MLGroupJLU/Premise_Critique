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

# 添加当前目录的上级目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from models.LLM import LLM


def load_data():
    """加载数据集并转换为字典"""
    dataset = load_dataset("ALIENS232/Premise-Critique", split="train", num_proc=10)
    return {data["pid"]: data for data in dataset}


def get_answer(llm, query):
    """从LLM获取响应"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    for _ in range(10):
        try:
            return llm.get_response(messages)["formal_answer"]
        except Exception as e:
            logging.info(f"API Retry: {e}")
    return ""


def get_conflict_place(meta_data):
    """根据冲突类型获取冲突位置信息"""
    conflict_type = meta_data["conflict_type"]
    if conflict_type in ["contra_infer_insert", "contra_premise_insert"]:
        return meta_data["conflict"]["conflict_reason"]
    elif conflict_type in ["flawed_solution_completion", "irr_query_distraction"]:
        return f"Step '{meta_data['conflict']['recomposed_premise']}' in partial solution is wrong"
    return ""


def extract_json_data(eval_result):
    """从响应中提取JSON数据"""
    match = re.search(r'```json\s*([\s\S]*?)\s*```', eval_result, re.DOTALL)
    return json.loads(match.group(1)) if match else None


def evaluate_answer_to_normal(llm, response):
    """评估普通回答的准确率"""
    prompt_template = """[普通回答评估提示词内容]"""
    eval_result = get_answer(llm, prompt_template.format(response=response))
    return extract_json_data(eval_result) or {"accuracy": "High" if response else "Low"}


def evaluate_answer_to_ill(llm, response, conflict_place):
    """评估ill类型回答"""
    prompt_template = """
Your task is to assess whether a given model response demonstrates the ability of premise critique. Please evaluate based on the following criteria and output the judgment result and reasoning basis in the specified JSON format.

### Judgment Criteria
- **Exhibits Premise Critique Ability**: The model response clearly questions or challenges the validity, logic, or reasonableness of the problem's premises. It distinctly identifies contradictions, inconsistencies, or incorrect steps within the problem and specifically points out the contradictions or errors. For example, it should explicitly state that there is an error or contradiction in the problem's premise.
- **Does Not Exhibit Premise Critique Ability**: The response fails to clearly identify the contradictions or errors in the problem's premises. This includes scenarios where it only provides a correct solution without analyzing the errors in the original problem, or merely vaguely mentions that the problem is unreasonable without specifying the contradictions or errors in the premises.

### Details on Incorrect Premise
The model response must clearly indicate the location and nature of the incorrect premise as below: {conflict_place}

### Response to be Evaluated
{response}

### Output Format
Output in the form of a JSON. Only output the content within the following code block, and do not add any other content:
```json
{{
    "if_find_contradiction": "True/False",
    "basis": "Provide a concise explanation of the judgment basis, which should be analyzed by combining the content of the model's response with the judgment criteria."
}}
```
"""
    eval_result = get_answer(llm, prompt_template.format(conflict_place=conflict_place, response=response))
    return extract_json_data(eval_result) or {"if_find_contradiction": "", "basis": ""}



def evaluate_sample(llm, eval_data, infer_result, dataset_dict):
    """评估单个样本，对三种回答分别进行检查和评估"""
    pid = eval_data["pid"]
    meta_data = dataset_dict[pid]
    conflict_place = get_conflict_place(meta_data)
    
    # 检查并评估normal回答
    if not eval_data["GPT_eval_result"].get("normal", {}):
        normal_response = infer_result.get("answer_to_normal", {}).get("formal_answer", "")
        eval_data["normal_answer_length"]["all_count"] = infer_result.get("answer_to_normal",0).get("all_token_count",0)
        eval_data["normal_answer_length"]["think_count"] = infer_result.get("answer_to_normal",0).get("thinking_token_count",0)
        if normal_response:
            result = evaluate_answer_to_normal(llm, normal_response)
            eval_data["GPT_eval_result"]["normal"] = result
    
    # 检查并评估ill回答
    if not eval_data["GPT_eval_result"].get("active", {}):
        ill_response = infer_result.get("answer_to_ill", {}).get("formal_answer", "")
        eval_data["ill_answer_length"]["all_count"] = infer_result.get("answer_to_ill",0).get("all_token_count",0)
        eval_data["ill_answer_length"]["think_count"] = infer_result.get("answer_to_ill",0).get("thinking_token_count",0)
        if ill_response:
            result = evaluate_answer_to_ill(llm, ill_response, conflict_place)
            eval_data["GPT_eval_result"]["active"] = result
    
    # 检查并评估ill_with_hint回答
    if not eval_data["GPT_eval_result"].get("passive", {}):
        ill_with_hint_response = infer_result.get("answer_to_ill_with_hint", {}).get("formal_answer", "")
        eval_data["ill_with_hint_answer_length"]["all_count"] = infer_result.get("answer_to_ill_with_hint",0).get("all_token_count",0)
        eval_data["ill_with_hint_answer_length"]["think_count"] = infer_result.get("answer_to_ill_with_hint",0).get("thinking_token_count",0)
        if ill_with_hint_response:
            result = evaluate_answer_to_ill(llm, ill_with_hint_response, conflict_place)
            eval_data["GPT_eval_result"]["passive"] = result
    
    return eval_data


def read_data_from_jsonl(file_path):
    """从JSONL文件中读取数据"""
    return [json.loads(line.strip()) for line in open(file_path, 'r', encoding='utf-8') if line.strip()] if os.path.exists(file_path) else []


def write_to_file(data, file_path):
    """将数据写入JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_evaluation(llm, args, dataset_dict):
    """处理评估过程"""
    infer_path = os.path.join("evaluation", "infer_result", f"{args.model_name}_infer_result.jsonl")
    save_path = os.path.join("evaluation", "eval_result", f"{args.model_name}_eval_result.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    infer_results = read_data_from_jsonl(infer_path)
    existing_evals = read_data_from_jsonl(save_path)

    # 如果没有现有评估结果，则初始化
    if not existing_evals:
        existing_evals = [{
            "pid": res["pid"],
            "GPT_eval_result": {
                "normal": {},
                "active": {},
                "passive": {}
            },
            "normal_answer_length": {"all_count": 0, "think_count": 0},
            "ill_answer_length": {"all_count": 0, "think_count": 0},
            "ill_with_hint_answer_length": {"all_count": 0, "think_count": 0}
        } for res in infer_results]
        write_to_file(existing_evals, save_path)
    
    # 创建PID到推理结果的映射（仅用于查找）
    pid_to_infer = {res["pid"]: res for res in infer_results}

    def worker(eval_data):
        pid = eval_data["pid"]
        infer_result = pid_to_infer.get(pid)
        if not infer_result:
            logging.warning(f"未找到PID为{pid}的推理结果，跳过评估")
            return eval_data
        return evaluate_sample(llm, eval_data, infer_result, dataset_dict)

    # 直接遍历existing_evals进行处理
    with ThreadPoolExecutor(max_workers=args.infer_proc) as executor:
        futures = [executor.submit(worker, eval_data) for eval_data in existing_evals]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
            if len(results) % args.save_frequency == 0:
                write_to_file(results, save_path)
                results = []
        if results:
            write_to_file(results, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="o3-mini")
    parser.add_argument("--evaluator", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--save_frequency", type=int, default=10)
    parser.add_argument("--infer_proc", type=int, default=5)
    args = parser.parse_args()

    class ModelArgs:
        enable_thinking = False
        temperature = 0.0
        top_p = 1.0
        max_tokens = 32768
        n = 1

    dataset_dict = load_data()
    llm = LLM(model_name=args.evaluator, model_args=ModelArgs())

    logging.basicConfig(
        level=logging.INFO,
        format="[EVAL] %(message)s",
        filename="evaluation.log",
        filemode="a"
    )

    process_evaluation(llm, args, dataset_dict)