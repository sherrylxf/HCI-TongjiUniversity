from flask import Flask, request, jsonify
from flask_cors import CORS
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
import os
from wsgiref.handlers import format_date_time
from time import mktime
import shutil
import glob
HISTORY_DIR = "history"  # 项目根目录下的保存路径
os.makedirs(HISTORY_DIR, exist_ok=True)
app = Flask(__name__)
CORS(app)
# 开放 history 目录供前端访问历史音频文件
app.static_folder = "history"
app.add_url_rule('/history/<path:filename>', endpoint='history', view_func=app.send_static_file)
# 配置讯飞 ISE 参数
HOST_URL = "ws://ise-api.xfyun.cn/v2/open-ise"
APPID = "5e11538f"
API_SECRET = "ff446b96b01252f80331ae6e4c64984a"
API_KEY = "91205afe0d17e38c61be35fca346503c"

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误", "detail": str(error)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "接口不存在"}), 404

def generate_ws_url():
    now = datetime.datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    signature_origin = f"host: ise-api.xfyun.cn\ndate: {date}\nGET /v2/open-ise HTTP/1.1"
    signature_sha = hmac.new(API_SECRET.encode(), signature_origin.encode(), digestmod=hashlib.sha256).digest()
    signature_base64 = base64.b64encode(signature_sha).decode()
    authorization = f"api_key=\"{API_KEY}\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"{signature_base64}\""
    authorization_base64 = base64.b64encode(authorization.encode()).decode()
    query = {
        "authorization": authorization_base64,
        "date": date,
        "host": "ise-api.xfyun.cn"
    }
    return f"{HOST_URL}?{urlencode(query)}"



def run_ise_evaluation(audio_path, text, save_basename=None):
    result = {"xml": None, "error": None}
    ws_url = generate_ws_url()
    final_result = {"xml": None}

    def on_message(ws, message):
        data = json.loads(message)
        if data["data"]["status"] == 2:
            xml = base64.b64decode(data["data"]["data"]).decode("utf-8")
            final_result["xml"] = xml
            ws.close()

    def on_error(ws, error):
        final_result["error"] = str(error)

    def on_open(ws):
        init_msg = {
            "common": {"app_id": APPID},
            "business": {
                "category": "read_sentence",
                "rstcd": "utf8",
                "sub": "ise",
                "group": "pupil",
                "ent": "en_vip",
                "tte": "utf-8",
                "cmd": "ssb",
                "auf": "audio/L16;rate=16000",
                "aue": "lame",
                "text": "\uFEFF" + text
            },
            "data": {"status": 0, "data": ""}
        }
        ws.send(json.dumps(init_msg))

        with open(audio_path, "rb") as f:
            while True:
                buf = f.read(1280)
                if not buf:
                    end_msg = {
                        "business": {"cmd": "auw", "aus": 4, "aue": "lame"},
                        "data": {"status": 2, "data": ""}
                    }
                    ws.send(json.dumps(end_msg))
                    break
                frame_msg = {
                    "business": {"cmd": "auw", "aus": 2, "aue": "lame"},
                    "data": {
                        "status": 1,
                        "data": base64.b64encode(buf).decode(),
                        "data_type": 1,
                        "encoding": "raw"
                    }
                }
                ws.send(json.dumps(frame_msg))
                time.sleep(0.04)

    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_open=on_open)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    if save_basename and final_result.get("xml"):
        # 保存音频和XML文件
        audio_target = os.path.join(HISTORY_DIR, f"{save_basename}.mp3")
        xml_target = os.path.join(HISTORY_DIR, f"{save_basename}.xml")
        shutil.copy(audio_path, audio_target)
        with open(xml_target, "w", encoding="utf-8") as f:
            f.write(final_result["xml"])

    return {"xml": final_result.get("xml"), "error": final_result.get("error")}

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    audio = request.files.get("audio")
    text = request.form.get("text")
    if not audio or not text:
        return jsonify({"error": "音频文件或文本缺失"}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '_')).strip().replace(" ", "_")
    save_basename = f"{timestamp}_{safe_text}"

    temp_path = f"./temp_{int(time.time())}.mp3"
    audio.save(temp_path)

    try:
        result = run_ise_evaluation(temp_path, text, save_basename=save_basename)
    finally:
        os.remove(temp_path)

    return jsonify(result)

from xml.etree import ElementTree as ET

@app.route("/api/history", methods=["GET"])
def get_history_records():
    import glob
    import xml.etree.ElementTree as ET
    import os

    records = []
    for mp3_path in glob.glob(os.path.join(HISTORY_DIR, "*.mp3")):
        base_name = os.path.splitext(os.path.basename(mp3_path))[0]
        xml_path = os.path.join(HISTORY_DIR, f"{base_name}.xml")

        total_score = "无"
        accuracy_score = "无"
        fluency_score = "无"
        standard_score = "无"
        text_full = ""
        rejected = False

        if os.path.exists(xml_path):
            try:
                with open(xml_path, "r", encoding="utf-8") as f:
                    xml_text = f.read()
                    xml_doc = ET.fromstring(xml_text)

                    # 尝试优先读取 read_chapter 节点
                    read_chapter = xml_doc.find(".//read_chapter")
                    read_sentence = xml_doc.find(".//read_sentence")

                    node = read_chapter if read_chapter is not None else read_sentence

                    if node is not None:
                        total_score_val = node.get("total_score")
                        accuracy_score_val = node.get("accuracy_score")
                        fluency_score_val = node.get("fluency_score")
                        standard_score_val = node.get("standard_score")

                        rejected = node.get("is_rejected") == "true"
                        text_full = node.get("content", "") or node.get("lan", "")

                        if total_score_val:
                            total_score_val = float(total_score_val)
                            total_score = f"{total_score_val:.2f} " if rejected else f"{total_score_val:.2f}"
                        if accuracy_score_val:
                            accuracy_score = f"{float(accuracy_score_val):.3f}"
                        if fluency_score_val:
                            fluency_score = f"{float(fluency_score_val):.3f}"
                        if standard_score_val:
                            standard_score = f"{float(standard_score_val):.3f}"

            except Exception as e:
                print(f"[XML解析失败] {xml_path} - {e}")

        stat = os.stat(mp3_path)
        records.append({
            "filename": f"{base_name}.mp3",
            "size": round(stat.st_size / 1024 / 1024, 2),  # MB
            "mtime": stat.st_mtime,
            "totalScore": total_score,
            "accuracyScore": accuracy_score,
            "fluencyScore": fluency_score,
            "standardScore": standard_score,
            "textFull": text_full,
            "textSnippet": (text_full[:20] + "…") if len(text_full) > 20 else text_full
        })

    # 按修改时间倒序排列
    records.sort(key=lambda r: r["mtime"], reverse=True)
    return jsonify(records)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
