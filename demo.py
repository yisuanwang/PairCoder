import os
def set_cuda_visible_devices(cuda_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


from api.chatglm_pro_api import ChatGLMProService
from api.chatglm_turbo_api import ChatGLM_TurboService
from api.minimax_api import MinimaxService
from api.baichuan2_7b_api import Baichuan2_7b_Service
from api.qwen_7b_api import Qwen_7b_chat_Service

class PairDevFramework:
    def __init__(self, driver, navigator, iscodeLLM=False):
        self.iscodeLLM = iscodeLLM
        # Initialize state
        self.default_driver_prompt = "You are now the driver in pair programming and your task is to write code. Please follow the instructions below to generate the code, and only return the full code content, not the extra text."
        self.default_navigator_prompt = "You are now the Navigator in pair programming and your task is to review the code and provide feedback. Please review the code below to indicate if there is an error, or just return [NOERROR] if there is no error."
        self.code = ""  # Current generated code
        self.errors = []  # Errors in the code
        self.driver = driver  # Initialize the driver as LLM1
        self.navigator = navigator  # Initialize the navigator as LLM2
        self.question = ""

    def reset(self):
        self.driver.clear_history()
        self.navigator.clear_history()
        self.code = ""  # Current generated code
        self.errors = []  # Errors in the code
        self.question = ""

    def switch_roles(self):
        # Switch roles
        self.driver, self.navigator = self.navigator, self.driver
        self.driver.make_request(self.default_driver_prompt)
        self.navigator.make_request(self.default_navigator_prompt)

    def generate_code(self, prompt):
        self.code = self.driver.make_request(prompt)
        print('='*25, 'code, driver=', self.driver.get_name(), '='*25, '\n', self.code)

    def review_code(self):
        # Code review method
        review_prompt = f"Review the code below to indicate if there is an error, and only return [NOERROR] if there is no error. The question is [{self.question}]. The code you need to check is [{self.code}]"
        review_results = self.navigator.make_request(review_prompt).replace('\"', '').replace('\\n', '\n')
        self.errors = review_results
        print('='*25, 'review_results, navigator=', self.navigator.get_name(), '='*25, '\n', review_results)

    def fix_errors(self):
        # Method to fix code errors
        fix_prompt = f"Follow the instructions below to fix errors in the code. Your answer only needs to contain the code. The question is [{self.question}]. The code you just generated is [{self.code}]. The review is [{self.errors}]"
        # self.generate_code(fix_prompt)
        self.code = self.driver.make_request(fix_prompt).replace('\"', '').replace('\\n', '\n')
        print('='*25, 'fix_errors, driver=', self.driver.get_name(), '='*25, '\n', self.code)

    def run(self, initial_prompt):
        self.question = initial_prompt
        # Initialize role responsibilities
        self.driver.make_request(self.default_driver_prompt)
        self.navigator.make_request(self.default_navigator_prompt)
        # Main run loop
        self.code = initial_prompt
        self.generate_code(initial_prompt)
        self.review_code()
        while not self.errors.__contains__('NOERROR'):
            self.fix_errors()
            self.review_code()
            self.switch_roles()  # Switch roles after each iteration to ensure participation from both sides
        return self.code

set_cuda_visible_devices("1,2,3,4")

# framework = PairDevFramework(driver = ChatGLMProService(), navigator = ChatGLMProService())

# framework = PairDevFramework(driver = MinimaxService(save_history=False), 
                            #  navigator = ChatGLMProService(save_history=False))

framework = PairDevFramework(driver = ChatGLM_TurboService(save_history=False), 
                                navigator = MinimaxService(save_history=False))

framework.reset()

initial_prompt = '''
def histogram(test):
    """Given a string representing a space separated lowercase letters, return a dictionary
    of the letter with the most repetition and containing the corresponding count.
    If several letters have the same occurrence, return all of them.

    Example:
    histogram('a b c') == {'a': 1, 'b': 1, 'c': 1}
    histogram('a b b a') == {'a': 2, 'b': 2}
    histogram('a b c a b') == {'a': 2, 'b': 2}
    histogram('b b b b a') == {'b': 4}
    histogram('') == {}

    """
'''

generated_code = framework.run(initial_prompt)
print(generated_code)
