
import requests,time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from transformers import pipeline
from transformers import AutoConfig, AutoModel
import requests
import torch

 
import os
def set_cuda_visible_devices(cuda_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices



class Baichuan2_7b_Service():
    model_name = "Baichuan2-7B-Chat"
    save_history = True

    def __init__(self, save_history = True) -> None:
        self.history = []
        self.save_history = save_history
        model_path = 'xxx/Baichuan2-7B-Chat'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        pass
    
    def make_request(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, self.history)
        self.history.append({"role": "assistant", "content": response})
        # 清楚历史记录
        if self.save_history:
            self.clear_history()
        return response

    def clear_history(self):
        self.history = []
        return 'clear_history success'

    def get_history(self):
        return self.history

    def get_name(self):
        return self.model_name
    
    pass

if __name__ == '__main__':
    
    # set_cuda_visible_devices('0')

    chatglmpro_service = Baichuan2_7b_Service()

    initial_prompt = '''from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''
    print(chatglmpro_service.make_request(initial_prompt))
