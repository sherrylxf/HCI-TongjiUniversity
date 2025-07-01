import gradio as gr
from llm_api import call_llm

# 保存多段历史对话记录
session_histories = {}


def chat(model_name, user_input, history):
    if history is None:
        history = []

    # 构建完整上下文对话
    messages = [{"role": "system", "content": "你是一个智能助手。"}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_input})

    # 调用支持上下文的模型接口
    reply = call_llm(model_name, messages)

    if "<think>" in reply and "</think>" in reply:
        reply = reply.split("</think>")[-1].strip()
    # 更新历史记录
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

    return history, history, ""


def save_history(history, name):
    if name:
        session_histories[name] = history
    return gr.update(choices=list(session_histories.keys())), []


# 读取已有历史记录
def load_history(name):
    return session_histories.get(name, [])


with gr.Blocks() as demo:
    gr.Markdown("# 智能聊天机器人")
    gr.Markdown("支持 DeepSeek-R1、GLM-4和Qwen模型。支持多轮对话和历史记录加载。")

    with gr.Row():
        model_select = gr.Dropdown(["DeepSeek-R1", "GLM-4", "Qwen"], label="选择模型", value="DeepSeek-R1")
        history_name = gr.Dropdown(label="选择历史记录对话", choices=[])

    chatbot = gr.Chatbot(type='messages',height=500)

    with gr.Row():
        msg = gr.Textbox(label="请输入你的问题", scale=4)
        submit_btn = gr.Button("发送", scale=1)

    with gr.Row():
        save_name = gr.Textbox(label="保存当前对话为（输入名称）")
        save_btn = gr.Button("保存对话")
        clear_btn = gr.Button("清空当前对话")

    # 设置提交事件
    msg.submit(chat, inputs=[model_select, msg, chatbot], outputs=[chatbot, chatbot, msg])
    submit_btn.click(chat, inputs=[model_select, msg, chatbot], outputs=[chatbot, chatbot, msg])

    save_btn.click(save_history, inputs=[chatbot, save_name], outputs=[history_name, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)
    history_name.change(load_history, inputs=history_name, outputs=chatbot)

demo.launch()