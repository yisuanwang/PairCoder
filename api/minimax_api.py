
import requests,time

class MinimaxService():
    model_name = "minimax-abab5-chat[api]"
    save_history = True

    def __init__(self, save_history = True) -> None:
        # your api key
        self.group_id = "xxx"
        self.api_key = "xxx.xxx.xxx-xxx-xxx-QxAwCP-xxx-ewh5fns6-xxx"
        self.history = []
        self.save_history = save_history
        pass

    def make_request(self, prompt):
        url = f'https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.group_id}'
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.history.append({"sender_type": "USER","text": prompt})
        request_body = {
            "model":"abab5-chat",
            "tokens_to_generate": 2048,
            'messages': self.history
        }
        response = requests.post(url, headers=headers, json=request_body)
        reply = response.json()['reply']
        
        self.history.append({"sender_type": "BOT","text": reply})
        
        if self.save_history:
            self.clear_history()
        return reply


    def clear_history(self):
        self.history = []
        return 'clear_history success'

    def get_history(self):
        return self.history

    def get_name(self):
        return self.model_name
    
    pass

if __name__ == '__main__':

    chatglmpro_service = MinimaxService()

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
