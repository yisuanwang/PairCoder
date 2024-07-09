import zhipuai
import requests,time

class ChatGLMProService():
    model_name = "ChatGLM-Pro[api]"
    save_history = True
    
    def __init__(self, save_history = True) -> None:
        # your api key
        zhipuai.api_key = "xxxxx"
        self.history = []
        self.save_history = save_history
        pass

    def make_request(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        # print('history=',self.history)
        response = zhipuai.model_api.invoke(
            model="chatglm_pro",
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

    chatglmpro_service = ChatGLMProService()

    print(chatglmpro_service.make_request('你好！我叫陈大帅逼，你叫什么？'))
    # print(chatglmpro_service.make_request('哦哦，你可以帮我以“陈大帅逼”为首字作一首藏头诗吗？'))
    # print(chatglmpro_service.make_request('我叫什么名字？'))
    # print(chatglmpro_service.clear_history())
    # print(chatglmpro_service.make_request('你知道我叫什么名字吗？'))