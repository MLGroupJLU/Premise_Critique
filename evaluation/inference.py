import os
import argparse
import concurrent
import tempfile
from datasets import load_dataset
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import re

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from models.LLM import LLM

def get_answer(llm, query):
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]
    for _ in range(0,10):
        try:
            return llm.get_response(messages)
        except Exception as e:
            logging.info(f"API failed! retrying! error: {e}")
    return ""
    

def answer_ill_query(llm, problem_data):
    # ill_query = problem_data["ill_query"] # 主动识别

    # ollama no_think
    # ill_query = '/think' + problem_data["ill_query"] # 主动识别
    ill_query = '/no_think' + problem_data["ill_query"] # 主动识别
    return get_answer(llm, ill_query)

def answer_normal_query(llm, problem_data):
    # normal_query = problem_data["normal_query"] # 正常回答

    # ollama no_think
    # normal_query = '/think' + problem_data["normal_query"] # 正常回答
    normal_query = '/no_think' + problem_data["normal_query"] # 正常回答
    return get_answer(llm, normal_query)

def answer_ill_query_with_hint(llm, problem_data):
    # ill_query = problem_data["ill_query"] # 被动识别

    # ollama no_think
    # ill_query = '/think' + problem_data["ill_query"] # 被动识别
    ill_query = '/no_think' + problem_data["ill_query"] # 被动识别
    ill_query += """
Check if there are any errors in the question's premises before answering. If there are, please report them promptly. 
"""
    return get_answer(llm, ill_query)

# ollama
def extract_after_think(text):
    # 找到最后一个 </think> 标签的位置
    matches = list(re.finditer(r"</think>", text))
    if matches:
        last_match = matches[-1]
        after_think = text[last_match.end():]
        return after_think.lstrip()  # 去除前导空白
    return text.strip()

def process_answer_dict(answer_dict):
    # 处理 formal_answer 字段，防止缺失或类型错误
    formal_answer = answer_dict.get("formal_answer", "")
    if not isinstance(formal_answer, str):
        formal_answer = ""  # 如果不是字符串，返回空字符串

    return {
        **answer_dict,
        "formal_answer": extract_after_think(formal_answer)
    }

def inference(llm, problem_data):
    # print(answer_normal_query(llm, problem_data))
    # inference_result_template = {
    #     "pid": problem_data["pid"],
    #     "answer_to_normal": answer_normal_query(llm, problem_data),
    #     "answer_to_ill": answer_ill_query(llm, problem_data),
    #     "answer_to_ill_with_hint": answer_ill_query_with_hint(llm, problem_data)
    # }

    # ollama
    inference_result_template = {
        "pid": problem_data["pid"],
        "answer_to_normal": process_answer_dict(answer_normal_query(llm, problem_data)),
        "answer_to_ill": process_answer_dict(answer_ill_query(llm, problem_data)),
        "answer_to_ill_with_hint": process_answer_dict(answer_ill_query_with_hint(llm, problem_data))
    }
    return inference_result_template

def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_dataset(llm, dataset, save_path, save_frequency, infer_proc):
    def process_problem(cur_data):
        return inference(llm, cur_data)
    
    inference_result_list = []
    try:
        with open(save_path,'r',encoding='utf-8') as f:
            result_lines = f.readlines()
            for line in result_lines:
                obj = json.loads(line.strip())
                inference_result_list.append(obj)
    except Exception as e:
        pass
    infered_pid_list=[result["pid"] for result in inference_result_list]
    filtered_dataset = [data for data in dataset if data["pid"] not in infered_pid_list] # 去掉已经推理过的

    with concurrent.futures.ThreadPoolExecutor(max_workers=infer_proc) as executor:
        futures = [executor.submit(process_problem, data) for data in filtered_dataset]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                inference_result_list.append(result)
                logging.info(f"pid:{result["pid"]} finished, evaled_len:{len(inference_result_list)}")
                if len(inference_result_list) % args.save_frequency == 0:
                    write_to_file(inference_result_list, save_path)
            except Exception as e:
                logging.error(f"error {e}")
                continue

    # 保存剩余的结果
    if inference_result_list:
        write_to_file(inference_result_list, save_path)

def check_data(llm, save_path, save_frequency):
    # 临时文件路径，在原文件名后添加 _temp 后缀
    temp_save_path = save_path + "_temp"
    # 开始检查数据
    with open(save_path, 'r', encoding='utf-8') as f:
        final_lines = f.readlines()

    checked_data = []
    fail_num = 0
    for line in final_lines:
        cur_data = json.loads(line.strip())
        if len(cur_data["answer_to_normal"]) == 0:
            logging.info(f"pid: {cur_data['pid']} fail")
            cur_data["answer_to_normal"] = answer_normal_query(llm, cur_data)
        if len(cur_data["answer_to_ill"]) == 0:
            logging.info(f"pid: {cur_data['pid']} fail")
            cur_data["answer_to_ill"] = answer_ill_query(llm, cur_data)
        if len(cur_data["answer_to_ill_with_hint"]) == 0:
            logging.info(f"pid: {cur_data['pid']} fail")
            cur_data["answer_to_ill_with_hint"] = answer_ill_query_with_hint(llm, cur_data)

        if len(cur_data["answer_to_normal"]) == 0 or len(cur_data["answer_to_ill"]) == 0 or len(
                cur_data["answer_to_ill_with_hint"]) == 0:
            fail_num += 1
        checked_data.append(cur_data)
        print(f"{len(checked_data)} samples checked!")
        if len(checked_data) % save_frequency == 0:
            write_to_file(checked_data, temp_save_path)

    if checked_data:
        write_to_file(checked_data, temp_save_path)
    logging.info(f"inference result checked! fail num: {fail_num}")

    # 全部检查结束后，将临时文件内容存回 save_path
    with open(temp_save_path, 'r', encoding='utf-8') as temp_f, open(save_path, 'w', encoding='utf-8') as f:
        f.write(temp_f.read())

    # 删除临时文件
    import os
    os.remove(temp_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval inference")
    parser.add_argument("--model_name", type=str, default="o4-mini-medium", help="the model which is used to be evaluated")
    parser.add_argument("--save_frequency", type=int, default=1, help="")
    parser.add_argument("--mode", type=str, default="inference", help="two modes: inference, check")
    parser.add_argument("--DEBUG", type=bool, default=True, help="")
    parser.add_argument("--dataset_load_proc", type=int, default=10, help="dataset load process")
    parser.add_argument("--infer_proc", type=int, default=1, help="model inference proc num")
    parser.add_argument("--stream",action='store_false', help="if use stream mode in API")
    # 与模型有关的参数，仅对开源模型生效
    parser.add_argument("--enable_thinking", action='store_true', help="model args")
    parser.add_argument("--thinking_budget", type=int, default=32768,help="model args")
    parser.add_argument("--temperature", type=float, default=0.0, help="model args")
    parser.add_argument("--top_p", type=float, default=1, help="model args")
    parser.add_argument("--max_tokens", type=int, default=32768, help="model args")
    args = parser.parse_args()

    class Model_AGRS:
        enable_thinking = False # 开启思考模式，这里设为 False 表示不开启
        temperature = 0.0 # temperature 设为 0 时，模型会进行贪婪解码，只选择概率最大的词
        top_p = 1 # top_p 设为 1 表示考虑所有可能的词，在贪婪解码中通常如此设置
        max_tokens = 32768 # 生成的最大 token 数量
        n=1 # 生成的文本序列数量
    
    model_args=Model_AGRS()
    model_args.stream = args.stream
    model_args.enable_thinking = args.enable_thinking
    model_args.thinking_budget=args.thinking_budget
    model_args.temperature = args.temperature
    model_args.top_p = args.top_p
    model_args.max_tokens = args.max_tokens

    # 模型名（主要针对qwen3）
    save_model_name=args.model_name
    if "/" in save_model_name:
        save_model_name = save_model_name.split("/")[-1]
    if model_args.enable_thinking:
        thinking_mode = "thinking"
        save_model_name=f"{args.model_name}_thinking"

    bench_data_path = os.path.join("evaluation", "dataset", "final_data.jsonl")
    save_path = os.path.join("evaluation", "infer_result", f"{save_model_name}_infer_result.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    llm = LLM(model_name=args.model_name,model_args=model_args)
    
    # 定义日志过滤函数
    def filter_log(record):
        unwanted_strings = ["HTTP Request", "HTTP/1.1"]
        message = record.getMessage()
        for s in unwanted_strings:
            if s in message:
                return False
        return True
    
    logging.basicConfig(
        level=logging.INFO,
        format="[eval inference] %(message)s",
        datefmt="[%X]",
        filename="premise_critique.log"
    )
    logger = logging.getLogger()
    # 为所有处理器添加过滤函数
    for handler in logger.handlers:
        handler.addFilter(filter_log)
    logging.info(f"running file {__file__}")

    if args.mode == "inference":
        dataset = load_dataset("ALIENS232/Premise-Critique", split="train", num_proc=args.dataset_load_proc)
        if args.DEBUG:
            dataset = dataset.select(range(1))

            # dataset = dataset.select([221])

        dataset_list = dataset.to_list()
        process_dataset(llm, dataset_list, save_path, args.save_frequency, args.infer_proc)
    else:
        check_data(llm, save_path, args.save_frequency)
    