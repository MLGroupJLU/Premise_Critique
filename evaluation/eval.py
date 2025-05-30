import os
import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import concurrent
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


def evaluate_answer_to_normal(llm, response,correct_answer=""):
    """评估普通回答的准确率"""
    prompt_template = """
Please act as an impartial judge to determine whether the final answer in the given response is correct, i.e., whether it aligns with the provided correct answer.
First, identify the final answer within the response.
Then, assess if it matches the correct answer, disregarding superficial formatting differences such as spacing, punctuation, capitalization, or structural presentation that do not affect the core content.

### Judgment Criteria
If the final answer is correct, output "True".
If the final answer is incorrect or cannot be found, output "False".

### Response to be evaluated
{response}

### Correct Answer
{correct_answer}

## Output Format
Only output string "True" or "False" without any additional content.
"""
    eval_output = get_answer(llm, prompt_template.format(response=response,correct_answer=correct_answer))
    if eval_output.lower()!="true" and eval_output.lower()!="false":
        return {}
    is_correct = str(eval_output).strip().lower() == "true"
    eval_result = {"correctness": "True" if is_correct else "False"}
    return eval_result


def evaluate_answer_to_ill(llm, response, conflict_place):
    """评估ill类型回答"""
    prompt_template = """
Please act as an impartial judge. Your task is to assess whether a given model response demonstrates the ability of **Premise Critique Ability**. Please follow the criteria below and output your judgment in the specified JSON format.

### Judgment Criteria
**Exhibits Premise Critique Ability**:
The response must:
- Clearly question/challenge the validity, logic, or reasonableness of the problem's premises.
- Specifically identify contradictions, inconsistencies, or flawed reasoning within the premises.
- Explicitly state the nature of the error (e.g., "The premise incorrectly assumes X," "There is a contradiction between Y and Z").
**Does Not Exhibit Premise Critique Ability**:
The response fails to meet the above criteria if it:
- Provides a correct solution without analyzing errors in the original premises.
- Vaguely claims the problem is "unreasonable" or "flawed" without specifying exact contradictions or errors.

### Details on Incorrect Premise
When a response engages in premise critique, it must precisely identify: \n
{conflict_place} \n
= The specific location or aspect within the problem's premises where the error lies

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
    eval_result=extract_json_data(eval_result)
    if eval_result:
        return eval_result
    else:
        return {}


def evaluate_sample(llm, eval_data, infer_result, dataset_dict, eval_counts):
    """评估单个样本，对三种回答分别进行检查和评估，并更新评估计数"""
    pid = eval_data["pid"]
    meta_data = dataset_dict[pid]
    conflict_place = get_conflict_place(meta_data)
    if_eval_flag = False
    fail_eval_flag = False

    # # 检查并评估normal回答
    # if dataset_dict[pid]["conflict_type"]!="irr_query_distraction":
    if not eval_data["GPT_eval_result"].get("normal", {}):
        if_eval_flag= True
        normal_response = infer_result.get("answer_to_normal", {}).get("formal_answer", "")
        eval_data["normal_answer_length"]["all_count"] = infer_result.get("answer_to_normal",0).get("all_token_count",0)
        eval_data["normal_answer_length"]["think_count"] = infer_result.get("answer_to_normal",0).get("thinking_token_count",0)
        # if normal_response:
        #     eval_counts["total_evaluated"] += 1
        #     result = evaluate_answer_to_normal(llm, normal_response)
        #     eval_data["GPT_eval_result"]["normal"] = result
        #     if not result:
        #         eval_counts["failed_evaluations"] += 1
        #         fail_eval_flag = True

    # 检查并评估ill回答
    if not eval_data["GPT_eval_result"].get("active", {}):
        if_eval_flag= True
        ill_response = infer_result.get("answer_to_ill", {}).get("formal_answer", "")
        eval_data["ill_answer_length"]["all_count"] = infer_result.get("answer_to_ill",0).get("all_token_count",0)
        eval_data["ill_answer_length"]["think_count"] = infer_result.get("answer_to_ill",0).get("thinking_token_count",0)
        if ill_response:
            eval_counts["total_evaluated"] += 1
            result = evaluate_answer_to_ill(llm, ill_response, conflict_place)
            eval_data["GPT_eval_result"]["active"] = result
            if not result or result.get("if_find_contradiction", "") == "":
                eval_counts["failed_evaluations"] += 1
                fail_eval_flag = True
    
    # 检查并评估ill_with_hint回答
    if not eval_data["GPT_eval_result"].get("passive", {}):
        if_eval_flag= True
        ill_with_hint_response = infer_result.get("answer_to_ill_with_hint", {}).get("formal_answer", "")
        eval_data["ill_with_hint_answer_length"]["all_count"] = infer_result.get("answer_to_ill_with_hint",0).get("all_token_count",0)
        eval_data["ill_with_hint_answer_length"]["think_count"] = infer_result.get("answer_to_ill_with_hint",0).get("thinking_token_count",0)
        if ill_with_hint_response:
            eval_counts["total_evaluated"] += 1
            result = evaluate_answer_to_ill(llm, ill_with_hint_response, conflict_place)
            eval_data["GPT_eval_result"]["passive"] = result
            if not result or result.get("if_find_contradiction", "") == "":
                eval_counts["failed_evaluations"] += 1
                fail_eval_flag = True
    
    if if_eval_flag:
        # 更新评估计数
        eval_counts["evaluated_sample_num"] += 1
    if fail_eval_flag:
        eval_counts["fail_sample_num"] += 1

    return eval_data


def read_data_from_jsonl(file_path):
    """从JSONL文件中读取数据"""
    return [json.loads(line.strip()) for line in open(file_path, 'r', encoding='utf-8') if line.strip()] if os.path.exists(file_path) else []


def write_to_file(data, file_path):
    """将数据写入JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_empty_eval_result_dict(pid):
    """获取空评估字典"""
    return {
        "pid": pid,
        "GPT_eval_result": {
            "normal": {},
            "active": {},
            "passive": {}
        },
        "normal_answer_length": {"all_count": 0, "think_count": 0},
        "ill_answer_length": {"all_count": 0, "think_count": 0},
        "ill_with_hint_answer_length": {"all_count": 0, "think_count": 0}
    }

def process_evaluation(llm, args, dataset_dict):
    """处理评估过程"""
    infer_path = os.path.join("evaluation", "infer_result", f"{args.model_name}_infer_result.jsonl")
    save_path = os.path.join("evaluation", "eval_result", f"{args.model_name}_eval_result.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    infer_results = read_data_from_jsonl(infer_path)
    existing_evals = read_data_from_jsonl(save_path)

    # 如果没有现有评估结果，则初始化
    if not existing_evals:
        existing_evals = [get_empty_eval_result_dict(res["pid"]) for res in infer_results]
        write_to_file(existing_evals, save_path)
    else:
        evaled_pid=set([data["pid"] for data in existing_evals])
        # 更新现有评估结果
        for res in infer_results:
            if res["pid"] not in evaled_pid:
                existing_evals.append(get_empty_eval_result_dict(res["pid"]))

    if args.DEBUG:
        existing_evals = existing_evals[:3]

    # 创建PID到推理结果的映射（仅用于查找）
    pid_to_infer = {res["pid"]: res for res in infer_results}

    # 创建共享的评估计数对象
    shared_counts = {
        "evaluated_sample_num":0,
        "fail_sample_num":0,
        "total_evaluated": 0, 
        "failed_evaluations": 0
    }

    def worker(eval_data):
        pid = eval_data["pid"]
        infer_result = pid_to_infer.get(pid)
        if not infer_result:
            logging.warning(f"未找到PID为{pid}的推理结果，跳过评估")
            return eval_data
        
        # 评估样本并更新计数
        updated_data = evaluate_sample(llm, eval_data, infer_result, dataset_dict, shared_counts)
        return updated_data

    # 直接遍历existing_evals进行处理
    results = []
    with ThreadPoolExecutor(max_workers=args.infer_proc) as executor:
        futures = [executor.submit(worker, eval_data) for eval_data in existing_evals]
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            try:
                result = future.result()
                results.append(result)
                
                # 定期保存结果
                if (i + 1) % args.save_frequency == 0:
                    write_to_file(results, save_path)
                    logging.info(f"已保存 {i + 1} 个评估结果")
                    
            except Exception as e:
                logging.error(f"处理评估时出错: {e}")
    
    # 保存最终结果
    write_to_file(results, save_path)
    
    # 输出评估统计信息
    logging.info(f"本轮评估完成:")
    logging.info(f"被评估样本数: {shared_counts['evaluated_sample_num']}")
    logging.info(f"评估失败样本数: {shared_counts['fail_sample_num']}")
    logging.info(f"总评估次数(API): {shared_counts['total_evaluated']}")
    logging.info(f"评估失败次数(API): {shared_counts['failed_evaluations']}")
    
    
    print(f"评估完成!")
    print(f"被评估样本数: {shared_counts['evaluated_sample_num']}")
    print(f"评估失败样本数: {shared_counts['fail_sample_num']}")
    print(f"总评估次数(API): {shared_counts['total_evaluated']}")
    print(f"评估失败次数(API): {shared_counts['failed_evaluations']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-235B-A22B_thinking")
    parser.add_argument("--evaluator", type=str, default="o3-mini-high")
    parser.add_argument("--save_frequency", type=int, default=1)
    parser.add_argument("--infer_proc", type=int, default=20)
    parser.add_argument("--DEBUG", type=bool, default=False, help="")
    args = parser.parse_args()
    
    class ModelArgs:
        enable_thinking = False
        temperature = 0.0
        top_p = 1.0
        max_tokens = 32768
        n = 1
        stream=False

    dataset_dict = load_data()
    llm = LLM(model_name=args.evaluator, model_args=ModelArgs())

    logging.basicConfig(
        level=logging.INFO,
        format="[EVAL] %(message)s",
        filename="premise_critique.log",
        filemode="a"
    )
    logging.info(f"running file {__file__} evaluating: {args.model_name}")
    process_evaluation(llm, args, dataset_dict)
    #print(f"token count: {llm.get_token_count()}")
