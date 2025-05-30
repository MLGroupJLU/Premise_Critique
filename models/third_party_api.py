from openai import OpenAI

class CloseSourceLLM:
    def __init__(self, model_name, model_args):
        self.model_name = model_name
        self.total_token_cnt = 0
        self.API_KEY = "" #!
        self.BASE_URL = "" #!
        self.model_args = model_args
        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.API_KEY,
        )
        self.completion_kwargs = {
            "model": self.model_name,
        }
        #启用流式传输
        if model_args.stream:
            self.completion_kwargs["stream"] = True
            self.completion_kwargs["stream_options"] = {
                "include_usage": True
            }
        if any(keyword in model_name.lower() for keyword in ['qwen3','R1']):
            self.completion_kwargs["extra_body"] = {
                "enable_thinking": model_args.enable_thinking,
                "thinking_budget": model_args.thinking_budget,
            }
        # 针对开源模型或可以设置的模型使用greedy解码
        if not any(keyword in model_name.lower() for keyword in ['o1', 'o3', 'o4', 'claude', 'gemini', 'gpt']):
            self.completion_kwargs["temperature"] = model_args.temperature
            self.completion_kwargs["top_p"] = model_args.top_p

    def get_response(self, messages):
        self.completion_kwargs["messages"] = messages
        completion = self.client.chat.completions.create(**self.completion_kwargs)
        return_template = {
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }

        if self.model_args.stream:
            #print("stream")
            # 流式传输就是返回内容分一次次传回
            last_chunk = None # 通过最后一个chunk记录token数量
            for chunk in completion:
                last_chunk = chunk
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    # 思考内容
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        return_template["thinking_content"] += delta.reasoning_content
                    # 收到content，开始进行回复
                    if hasattr(delta, "content") and delta.content:
                        return_template["formal_answer"] += delta.content
            print(last_chunk)
            return_template["all_token_count"] = last_chunk.usage.completion_tokens
            if hasattr(last_chunk.usage.completion_tokens_details, 'reasoning_tokens'):
                return_template["thinking_token_count"] = last_chunk.usage.completion_tokens_details.reasoning_tokens

        else:
            #print("not stream")
            return_template["all_token_count"] = completion.usage.completion_tokens
            return_template["formal_answer"] = completion.choices[0].message.content.strip()
            try:
                if hasattr(completion.choices[0].message, 'reasoning'):
                    return_template["thinking_token_count"] = completion.choices[0].message.reasoning.strip()
                if hasattr(completion.usage.completion_tokens_details, 'reasoning_tokens'):
                    return_template["thinking_token_count"] = completion.usage.completion_tokens_details.reasoning_tokens
            except Exception as e:
                pass

        self.total_token_cnt += return_template["all_token_count"]
        return return_template
    
    def get_token_count(self):
        return self.total_token_cnt