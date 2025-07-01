# tts_synthesizer.py

import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

class Ws_Param:
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "aue": "raw",  # 可改为 "lame" 获取 MP3 格式
            "auf": "audio/L16;rate=16000",
            "vcn": "x4_xiaoyan",  # 替换为你要的发音人
            "tte": "utf8"
        }
        self.Data = {
            "status": 2,
            "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")
        }

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = f"host: ws-api.xfyun.cn\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'),
                                 signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(signature_sha).decode('utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

        params = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }

        return url + '?' + urlencode(params)


def synthesize_speech(text, output_path='./output.mp3',
                      appid='174cd4bd', apikey='ca8ae3526b434b8b042d6a8be6fbb298', apisecret='MGJmODEwZjdlZWE5NDYwMjM4M2I2ZDM0'):
    wsParam = Ws_Param(appid, apikey, apisecret, text)
    # 修改 BusinessArgs 为返回 MP3 格式
    wsParam.BusinessArgs["aue"] = "lame"  # lame 表示 MP3 编码

    ws_url = wsParam.create_url()
    audio_chunks = []

    def on_message(ws, message):
        try:
            message = json.loads(message)
            code = message.get("code", -1)
            if code != 0:
                print(f"[错误] 讯飞返回错误：{message.get('message')} (code: {code})")
                ws.close()
                return
            audio = base64.b64decode(message["data"]["audio"])
            audio_chunks.append(audio)
            if message["data"]["status"] == 2:
                ws.close()
        except Exception as e:
            print("[异常] 解析讯飞返回数据失败：", e)
            ws.close()

    def on_error(ws, error):
        print("[WebSocket错误]", error)

    def on_close(ws, close_status_code, close_msg):
        print("[WebSocket连接关闭]")
        if close_status_code or close_msg:
            print(f"[关闭信息] code={close_status_code}, message={close_msg}")
        if audio_chunks:
            with open(output_path, 'wb') as f:
                for chunk in audio_chunks:
                    f.write(chunk)
            print(f"[保存成功] 合成语音已保存到 {output_path}")

    def on_open(ws):
        def run(*args):
            data = {
                "common": wsParam.CommonArgs,
                "business": wsParam.BusinessArgs,
                "data": wsParam.Data
            }
            ws.send(json.dumps(data))
        thread.start_new_thread(run, ())

    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    return output_path if os.path.exists(output_path) else None
