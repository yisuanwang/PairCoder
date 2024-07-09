import zhipuai
import requests,time

class ChatGLM_TurboService():
    model_name = "ChatGLM_Turbo[api]"
    save_history = True
    
    def __init__(self, save_history = True) -> None:
        # your api key
        zhipuai.api_key = "xxxxxxxxx"
        self.history = []
        self.save_history = save_history
        pass

    def make_request(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        # print('history=',self.history)
        response = zhipuai.model_api.invoke(
            model="chatglm_turbo",
            prompt=self.history,
            temperature=0.95,
            top_p=0.7,
            incremental=True
        )
        # print('response=', response)
        self.history.append(response['data']['choices'][0])
        if self.save_history:
            self.clear_history()
        return response['data']['choices'][0]['content'][2:-1].replace('\\n','\n').replace('\\"','\"')

    def clear_history(self):
        self.history = []
        return 'clear_history success'

    def get_history(self):
        return self.history
    
    def get_name(self):
        return self.model_name
    pass

if __name__ == '__main__':

    chatglmpro_service = ChatGLM_TurboService()
