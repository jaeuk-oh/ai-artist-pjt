"""Gradio 데모 UI."""

import os
import uuid
import httpx
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def chat(message: str, history: list, fan_name: str, session_id: str):
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        resp = httpx.post(
            f"{API_BASE}/chat",
            json={"message": message, "session_id": session_id, "fan_name": fan_name},
            timeout=30.0,
        )
        reply = resp.json()["response"]
    except Exception as e:
        reply = f"연결 오류: {e}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, session_id


with gr.Blocks(title="ArtistMind - YURI") as demo:
    gr.Markdown("# 💖 ArtistMind — YURI (NOVA)\nK-Pop AI 페르소나 챗봇 MVP")

    session_state = gr.State("")
    fan_name = gr.Textbox(label="내 이름 (선택)", placeholder="예: 민지", scale=1)
    chatbot = gr.Chatbot(type="messages", height=480)
    msg = gr.Textbox(placeholder="유리에게 메시지를 보내보세요! 💌", show_label=False)

    msg.submit(chat, [msg, chatbot, fan_name, session_state], [chatbot, session_state]).then(
        lambda: "", outputs=msg
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
