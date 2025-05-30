from modelscope import AutoProcessor, Gemma3ForConditionalGeneration
import torch

class gemma3_27b:
    def __init__(self, model_name,model_args):
        self.model_args = model_args
        self.model_name = "LLM-Research/gemma-3-27b-it"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
        )

    def get_response(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.model_args.max_tokens,
            )
            generation = generation[0][input_len:]

        return_template={
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }

        return_template["all_token_count"] = len(generation)
        return_template["formal_answer"] = self.processor.decode(generation, skip_special_tokens=True)
        
        return return_template
