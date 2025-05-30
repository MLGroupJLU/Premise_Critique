from modelscope import AutoProcessor, Llama4ForConditionalGeneration
import torch

class llama4_maverick_17b:
    def __init__(self, model_name,model_args):
        self.model_args = model_args
        self.model_name = "LLM-Research/Llama-4-Maverick-17B-128E-Instruct"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name,
            attn_implementation="flex_attention",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def get_response(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 获取输入 token 长度
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.model_args.max_tokens,
        )

        return_template={
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }

        return_template["all_token_count"]=outputs[:, input_len:].shape[-1]
        return_template["formal_answer"] = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        return return_template
