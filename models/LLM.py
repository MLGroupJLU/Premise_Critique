import importlib
from models import *

class LLM():
    def __init__(self,model_name,model_args):
        self.model_name=model_name
        self.model=self._get_model(model_name,model_args)

    def _get_model(self,model_name,model_args):
        if any(keyword in model_name.lower() for keyword in ['claude', 'gemini','gpt','doubao','deepseek','o3','o4','o1','deepseek-r1','qwen3','deepseek-v3','llama-4','gemma','phi-4']):
            module = importlib.import_module(f"models.third_party_api")
            model_class = getattr(module, 'CloseSourceLLM')
            return model_class(model_name,model_args)
        else:
            module = importlib.import_module(f'models.{model_name}')
            model_class = getattr(module, model_name)
            return model_class(model_name,model_args)

    def get_response(self,messages):
        return self.model.get_response(messages)

    def get_token_count(self):
        return self.model.get_token_count()