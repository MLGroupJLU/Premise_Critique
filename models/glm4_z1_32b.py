from transformers import AutoModelForCausalLM, AutoTokenizer

class glm4_z1_32b:
    def __init__(self, model_name,model_args):
        self.model_args = model_args
        self.model_name = "THUDM/GLM-4-Z1-32B-0414"
        self.processor = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
        )

    def get_response(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)


        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": self.model_args.max_tokens,
        }

        out = self.model.generate(**generate_kwargs)

        return_template={
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }


        return_template["all_token_count"]=out[0][inputs["input_ids"].shape[1]:].shape[-1] # 计算总token数量（包括思考和回答）
        generate_resp = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:-1], skip_special_tokens=False)

        return_template["formal_answer"] = generate_resp.strip()
        return return_template
