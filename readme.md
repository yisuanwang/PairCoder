# PairCoder

## Run code
```bash
# generate
python demo.py
```
If you want to test PairCoder on Colab, please use paircoder.ipynb. Enter your OpenAI API key into it, and you will be able to test it on HumanEval.
 
## ⚠️Notice
Some models require an api key, such as chatGPT, ChatGLM and Minimax models. We give some experimental data for PairCoder in data/sample_result.txt

Table A. Comparison with baseline on HumanEval (pass@1).
| Model              | based on GPT-4-turbo | based on GPT-3.5-turbo |
|----------------------------------|---------|---------|
| single model                     | 86.6    | 69.51   |
| AgentCoder [1]                     | 89.6    | 79.9    |
| Self-collaboration [2]            | 90.2  | 74.4    |
| MetaGPT [3]                         | 85.9    | 62.8    |
| AgentVerse [4]                      | 89.0      | 75.6    |
| ChatDev [5]                         | 84.1    | 61.8   |
| Reflexion [6]                       | 91.0      | 68.1    |
| AGILECODER [7]                     | 90.9   | 70.5   |
| ours(PairCoder)                  | 91.03   | 81.16   |

---

Table B. Test results of PairCoder as an agent for code generation subtasks in large-scale automated software construction systems on HumanEval (pass@1). The number in parentheses indicates the accuracy rate when completing coding subtasks with Paircoder.

| Model              | based on GPT-4-turbo (Using PairCoder for Code Generation Subtasks)| based on GPT-3.5-turbo (Using PairCoder for Code Generation Subtasks) |
|----------------------------------|---------|---------|
| single model                     | 86.6    | 69.51   |
| Self-collaboration [2]              | 90.2 (93.9)   | 74.4 (79.9)   |
| MetaGPT [3]                         | 85.9 (87.8)   | 62.8 (72.0)   |
| ChatDev [5]                         | 84.1 (86.6)   | 61.8 (70.7)  |
| ours(PairCoder)                  | 91.03         | 81.16        |

```
[1] Huang, Dong, et al. "Agentcoder: Multi-agent-based code generation with iterative testing and optimisation." arXiv preprint arXiv:2312.13010 (2023).

[2] Dong, Yihong, et al. "Self-collaboration Code Generation via ChatGPT." ACM Transactions on Software Engineering and Methodology.

[3] Hong, Sirui, et al. "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." The Twelfth International Conference on Learning Representations.

[4] Chen, Weize, et al. "Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors." The Twelfth International Conference on Learning Representations. 2023.

[5] Qian, Chen, et al. "Communicative agents for software development." arXiv preprint arXiv:2307.07924 (2023).

[6] Shinn, Noah, et al. "Reflexion: Language agents with verbal reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).

[7] Nguyen, Minh Huynh, et al. "AgileCoder: Dynamic Collaborative Agents for Software Development based on Agile Methodology." arXiv preprint arXiv:2406.11912 (2024).
```

