import time
from dataclasses import dataclass


import os
from openai import OpenAI



class openai_Service():
    model_name = "gpt-3.5-turbo"
    save_history = True

    def __init__(self, save_history = True, model_name = "gpt-3.5-turbo") -> None:
        self.history = []
        self.save_history = save_history
        self.model_name = model_name
        self.api_key = "sk-xxx"
        pass

    def make_request(self, prompt):
        client = OpenAI(api_key=self.api_key)
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
            )
            assistant_message = chat_completion.choices[0].message.content
            return assistant_message
        except Exception as e:
            print(f"An error occurred: {e}")
        

    def clear_history(self):
        self.history = []
        return 'clear_history success'

    def get_history(self):
        return self.history

    def get_name(self):
        return self.model_name
    
    pass

if __name__ == '__main__':

    service = openai_Service(model_name = "gpt-3.5-turbo")

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

    print(service.make_request(initial_prompt))

