from transformers import AutoModelForCausalLM, AutoTokenizer

class qwq_32b:
    def __init__(self, model_name,model_args):
        self.model_args = model_args
        self.model_name = "Qwen/QwQ-32B"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def get_response(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.model_args.max_tokens,
        )

        # 截取新生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return_template={
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }

        return_template["all_token_count"] = len(generated_ids[0]) # 计算总token数量（包括思考和回答）
        return_template["formal_answer"] = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return return_template
