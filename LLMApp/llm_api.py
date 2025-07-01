import requests
from openai import OpenAI
import httpx  # 手动引入 httpx 客户端

# 手动构建不带 proxies 的 httpx client
http_client = httpx.Client(
    base_url="https://llmapi.tongji.edu.cn/v1",  # 设置 base_url
    timeout=60.0,
    follow_redirects=True
)

# 初始化 OpenAI 客户端，传入显式 http_client
tongji_client = OpenAI(
    api_key="hCYuROnrEwQZlLK440E1B19dFa584fD2AdDf6a780d78C31c",
    http_client=http_client
)
# 智谱 GLM-4 API
glm_api_key = "cb1c1ce7f5e049aaabab14557774edcb.T9jwg6x5qk838Weq"
glm_api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 通义千问 Qwen API
qwen_api_key = "sk-6208fc97a97c484fab441c30a0eaa0c1"
qwen_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def call_llm(model_name, messages):
    model_name = model_name.lower()

    try:
        if model_name == "deepseek-r1":
            try:
                response = tongji_client.chat.completions.create(
                    model="DeepSeek-R1",
                    messages=messages
                )
                if hasattr(response.choices[0].message, "content"):
                    return response.choices[0].message.content.strip()
                else:
                    return "DeepSeek-R1 返回了空内容。"
            except Exception as e:
                return f"DeepSeek-R1 调用失败: {str(e)}"

        elif model_name == "glm-4":
            headers = {
                "Authorization": f"Bearer {glm_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "glm-4",
                "messages": messages,
                "temperature": 0.95,
                "top_p": 0.7,
                "stream": False
            }
            response = requests.post(glm_api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()

        elif model_name == "qwen":
            headers = {
                "Authorization": f"Bearer {qwen_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "qwen-plus",
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            }
            response = requests.post(qwen_api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['output']['text'].strip()

        else:
            return "暂不支持该模型。"

    except Exception as e:
        print(f"接口调用异常: {str(e)}")
        return f"接口调用异常: {str(e)}"
