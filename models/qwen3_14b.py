from transformers import AutoModelForCausalLM, AutoTokenizer

class qwen3_14b:
    def __init__(self, model_name,model_args):
        self.model_args = model_args
        self.model_name = "Qwen/Qwen3-14B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def get_response(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.model_args.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.model_args.max_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        return_template={
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }
        # 计算总token数量（包括思考和回答）
        return_template["all_token_count"] = len(output_ids)
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
            return_template["thinking_token_count"] = len(output_ids[:index]) # 思考token数量
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n") # 思考内容
        except ValueError:
            index = 0

        return_template["formal_answer"] = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # 添加标签
        return_template["thinking_content"] = f"<think>{thinking_content}</think>"
        # 拼接内容
        return return_template
