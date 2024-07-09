import dashscope
import requests,time
import random
from dashscope import Generation
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from transformers import pipeline
from transformers import AutoConfig, AutoModel
import requests
import torch

 
import os
def set_cuda_visible_devices(cuda_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices



class Qwen_7b_chat_Service():
    model_name = "qwen-7b-chat"
    save_history = True

    def __init__(self, save_history = True) -> None:
        self.history = []
        self.save_history = save_history
        dashscope.api_key = "sk-xxx"
        pass

    def make_request(self, prompt):
        gen = Generation()
        response = gen.call(
            'qwen-7b-chat',
            messages=prompt,
            seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
            result_format='message',  # set the result to be "message" format.
        )
        print(response)
        return response.output.choices[0].message.content

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

    chatglmpro_service = Qwen_7b_chat_Service()

#     initial_prompt = '''from typing import List
# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     """ Check if in given list of numbers, are any two numbers closer to each other than
#     given threshold.
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     True
#     """
# '''
#     print(chatglmpro_service.make_request(initial_prompt))

