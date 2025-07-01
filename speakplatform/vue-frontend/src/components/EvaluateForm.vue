<template>
  <div class="page-wrapper">
<div class="page-header">
  <div class="header-left">
    <button class="action-btn back-btn" @click="$emit('back')">返回首页</button>
  </div>
  <h2 class="page-title">🎙️英语口语练习平台</h2>
</div>


    <div class="evaluate-form">    
    <form @submit.prevent="submitForm">
      <label for="text" class="left-label">📖 目标朗读文本（可输入修改）：</label>
      <textarea id="text" v-model="text" rows="4" required></textarea>

      <div class="upload-row">
        <label for="audio">🎧 上传语音文件（MP3，16kHz）：</label>
        <input id="audio" type="file" @change="handleFileUpload" accept=".mp3" required>
      </div>
  
      <div class="recorder">
        <p class="or-label">🎙️ 或直接录制音频：</p>

        <div class="recorder-buttons">
          <!-- 录音按钮 -->
          <button type="button":class="['action-btn', isRecording ? 'stop-btn' : 'record-btn']" @click="toggleRecording">
            {{ isRecording ? '🛑 停止录音' : '🎙️ 开始录音' }}
          </button>

          <div v-if="isRecording" class="waveform">
           <span v-for="n in 10" :key="n" class="bar"></span>
          </div>
                  
         <!-- 回放按钮 -->
         <button v-if="audioBlob" type="button" class="action-btn replay-btn" @click="playAudio">
           ▶️ 回放录音
          </button>

         <!-- 🎵 音频播放器 -->
          <audio v-if="audioUrl" :src="audioUrl" ref="audioPlayer" controls class="audio-player"></audio>
          </div>
        </div>

      <!-- 提交按钮 -->
      <div class="submit-row">
        <button type="submit" :disabled="loading" class="action-btn submit-btn">
          {{ loading ? '提交中...' : '提交评测' }}
        </button>
      </div>
    </form>

    <div v-if="resultData" class="result-card">
      <h3>📊 评测结果</h3>
      <ul>
        <li><strong>总分：</strong>{{ resultData.total_score.toFixed(2) }}</li>
        <li><strong>准确度：</strong>{{ resultData.accuracy_score.toFixed(2) }}</li>
        <li><strong>流利度：</strong>{{ resultData.fluency_score.toFixed(2) }}</li>
        <li><strong>标准度：</strong>{{ resultData.standard_score.toFixed(2) }}</li>
      </ul>

      <!-- ▶️ 播放标准语音 -->
      <div v-if="ttsAudioUrl" class="tts-player">
        <h4>🎧 标准朗读示范：</h4>
        <audio :src="ttsAudioUrl" controls></audio>
      </div>

      <!-- 📌 低分提醒 -->
      <div v-if="lowScoreWords.length > 0" class="warning-box">
        <h4>⚠️ 以下单词得分较低，请注意发音：</h4>
        <ul>
          <li v-for="(item, index) in lowScoreWords" :key="index">
            <strong>{{ item.word }}</strong>：{{ item.score.toFixed(2) }} 分
          </li>
        </ul>
      </div>
    </div>    
  </div>
  </div>
  
</template>

<script>
import { ref } from "vue";
import Recorder from "recorder-core";
import "recorder-core/src/engine/mp3"; 
import "recorder-core/src/engine/mp3-engine";

export default {
  setup() {
    const text = ref("She is reading a book under the tree.");
    const audioFile = ref(null);
    const resultData = ref(null);
    const resultRaw = ref("");
    const loading = ref(false);
    const lowScoreWords = ref([]); 
    const isRecording = ref(false);
    const mediaRecorder = ref(null);
    const audioChunks = ref([]);
    const audioBlob = ref(null);
    const audioUrl = ref(null);
    const audioPlayer = ref(null);
    const ttsAudioUrl = ref(null);

    const handleFileUpload = (event) => {
      audioFile.value = event.target.files[0];
      console.log("上传文件:", audioFile.value);
    };

    let rec = null;

    const startRecording = () => {
      rec = Recorder({
        type: "mp3",
        sampleRate: 16000, // 讯飞要求16kHz
        bitRate: 16,
        audioTrackSet: {sampleRate:16000, channelCount:1}
      });

      rec.open(() => {
        // 清空旧的回放数据
        audioBlob.value = null;
        audioUrl.value = null;

        rec.start();
        isRecording.value = true;
      }, (errMsg) => {
        alert("录音失败：" + errMsg);
      });
    };

    const stopRecording = () => {
      rec.stop((blob, duration) => {
        console.log("录音完成", blob, duration);
        audioBlob.value = blob;
        audioFile.value = new File([blob], "recorded_audio.mp3", { type: "audio/mp3" });
        isRecording.value = false;
        console.log("blob", blob);
        console.log("设置 audioFile", audioFile.value);
        submitForm(); // 录音完成直接调用
      }, (errMsg) => {
       alert("停止失败：" + errMsg);
     });
    };

    const toggleRecording = () => {
      if (isRecording.value) {
        stopRecording();
      } else {
        startRecording();
      }
    };

    const playAudio = () => {
      if (audioBlob.value) {
        audioUrl.value = URL.createObjectURL(audioBlob.value);
      }
    };

    const submitForm = async () => {
      console.log("提交前 audioFile:", audioFile.value);
      if (!audioFile.value) {
        resultRaw.value = "❌ 请选择音频文件";
        return;
      }

      lowScoreWords.value = [];
      const formData = new FormData();
      formData.append("text", text.value);
      formData.append("audio", audioFile.value);

      loading.value = true;
      resultData.value = null;
      resultRaw.value = "";

      try {
        const res = await fetch("http://localhost:5000/api/evaluate", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`${res.status} ${res.statusText}: ${errorText}`);
        }

        const ttsForm = new FormData();
        ttsForm.append("text", text.value);
        const ttsRes = await fetch("http://localhost:5000/api/tts", {
          method: "POST",
          body: ttsForm,
        });

        const ttsJson = await ttsRes.json();
        if (ttsJson.success) {
         ttsAudioUrl.value = "http://localhost:5000" + ttsJson.path;
        } else {
          console.warn("TTS 合成失败", ttsJson);
        }

        const data = await res.json();
        console.log("返回数据:", data);
        resultRaw.value = JSON.stringify(data, null, 2);
        console.log("原始XML：", resultRaw.value);

        // 从返回的 XML 里解析出核心分数
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(data.xml, "text/xml");
        const sentenceNode = xmlDoc.querySelector("sentence");

        if (!sentenceNode) {
          resultRaw.value = "❌ 解析XML失败，未找到<sentence>节点";
          return;
        }

        if (sentenceNode) {
          resultData.value = {
            total_score: parseFloat(sentenceNode.getAttribute("total_score")),
            accuracy_score: parseFloat(sentenceNode.getAttribute("accuracy_score")),
            fluency_score: parseFloat(sentenceNode.getAttribute("fluency_score")),
            standard_score: parseFloat(sentenceNode.getAttribute("standard_score")),
          };

          // 📌 提取单词得分
          const wordNodes = xmlDoc.querySelectorAll("sentence > word");
          wordNodes.forEach((word) => {
            const wordContent = word.getAttribute("content");
            const wordScore = parseFloat(word.getAttribute("total_score"));
            if (wordScore < 4.0) {
              lowScoreWords.value.push({ word: wordContent, score: wordScore });
            }
          });
        }

      } catch (err) {
        resultRaw.value = "❌ 请求失败：" + err.message;
      } finally {
        loading.value = false;
      }
    };

    return {
      text,
      audioFile,
      resultData,
      resultRaw,
      loading,
      lowScoreWords,
      handleFileUpload,
      submitForm,
      isRecording,
      toggleRecording,
      audioBlob,
      audioUrl,
      audioPlayer,
      playAudio,
      ttsAudioUrl,
    };
  },
};
</script>

<style scoped>
html, body {
  margin: 0;
  padding: 0;
  min-height: 100vh;
  overflow-x: hidden;
  overflow-y: auto;
}
/* 顶部标题固定 */
.page-header {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 60px;
  background: rgba(255, 255, 255, 0.5); /* 半透明背景 */
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
}

.page-title {
  color: rgb(0, 0, 0);
  font-size: clamp(22px, 2.5vw, 32px);
  margin: 0;
  white-space: nowrap;
}

.header-left {
  position: absolute;
  left: 1em;
  top: 50%;
  transform: translateY(-50%);
}


/* 页面主容器 */
.page-wrapper {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 40px;
  box-sizing: border-box;
  position: relative;
  z-index: 0;
}

/* 背景图层 */
.page-wrapper::before {
  content: "";
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: url("/background.jpg") center/cover no-repeat;
  background-attachment: fixed;
  z-index: -1;
}

/* 白色模糊卡片 */
.evaluate-form {
  width: 90vw;
  max-width: 1000px;
  background: rgba(255, 255, 255, 0.25);
  border: 1px solid rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  padding: min(40px, 5vw);
  border-radius: 20px;
  margin-bottom: 60px;
  /* 不要限制 max-height 否则内容可能看不到 */
  box-sizing: border-box;
}

h2 {
  text-align: center;
  font-size: clamp(24px, 2.5vw, 32px);
  margin-bottom: 3vh;
  color: #333;
}

label {
  display: block;
  margin-top: 2vh;
  font-weight: bold;
  color: #333;
  font-size: clamp(14px, 1.2vw, 18px);
}

.left-label {
  text-align: left;
  margin-left: 0;
  display: block;
}

textarea,
input[type="file"] {
  width: 100%;
  padding: 1em;
  margin-top: 1vh;
  margin-bottom: 2vh;
  border: none;
  font-size: clamp(14px, 1.1vw, 16px);
  box-sizing: border-box;
}

.recorder {
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 12px;
  margin-top: 1vh;
  margin-bottom: 2vh;
}

.or-label {
  margin: 0;
  font-weight: bold;
  color: #333;
  font-size: clamp(14px, 1.2vw, 18px);
}

.recorder-buttons {
  display: flex;
  flex-direction: row;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}

.recorder span {
  color: #409eff;
  font-weight: bold;
}

audio {
  width: 100%;
  margin-top: 1.2vh;
  border-radius: 6px;
}

.result-card {
  margin-top: 4vh;
  padding: 3vh 2vw;
  background: #f9f9f9;
  border: 1px solid #ddd;
  border-radius: 10px;
}

.result-card h3 {
  margin-bottom: 2vh;
  font-size: clamp(18px, 1.5vw, 22px);
}

.result-card ul {
  padding-left: 0;
  list-style: none;
}

.result-card li {
  font-size: clamp(14px, 1.1vw, 16px);
  margin-bottom: 1vh;
}

.warning-box {
  margin-top: 2vh;
  padding: 2vh 2vw;
  background: #fff5f5;
  border-left: 5px solid #f56c6c;
  border-radius: 6px;
}

.warning-box h4 {
  color: #d93026;
  margin-bottom: 1vh;
  font-size: clamp(16px, 1.3vw, 18px);
}

.warning-box li {
  font-size: clamp(14px, 1.1vw, 15px);
  margin-bottom: 0.5vh;
}

pre {
  background: #f0f0f0;
  padding: 1em;
  border-radius: 6px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: clamp(13px, 1vw, 15px);
}

.upload-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 2vh;
  margin-bottom: 2vh;
}

.upload-row label {
  flex-shrink: 0;
  font-weight: bold;
  color: #333;
  font-size: clamp(14px, 1.2vw, 18px);
  margin: 0;
}

.upload-row input[type="file"] {
  flex-grow: 1;
  margin: 0;
  padding: 0.5em;
}

.submit-row {
  display: flex;
  justify-content: center;
  margin-top: 3vh;
}

/* 公共按钮样式 */
.action-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0.8em 1.6em;
  font-size: clamp(14px, 1vw, 16px);
  font-weight: bold;
  background: linear-gradient(135deg, #5ee7df 0%, #90abca 100%);
  border: none;
  border-radius: 10px;
  color: rgb(0, 0, 0);
  cursor: pointer;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
  transition: transform 0.2s ease, box-shadow 0.3s ease;
}

.action-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 14px rgba(0, 0, 0, 0.2);
}

.action-btn:active {
  transform: scale(0.97);
}

.action-btn:disabled {
  background: #b1c9d4;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}

/* 不同功能按钮颜色 */
.back-btn {
  background: linear-gradient(135deg, #ffffff 0%, #9ad7ff 100%);
}

.record-btn {
  background: linear-gradient(135deg, #CEE1FD 0%, #ffffff 100%);
}

.stop-btn {
  background: linear-gradient(135deg, #f85032 0%, #f8f8f8 100%);
}

.replay-btn {
  background: linear-gradient(135deg, #CEE1FD 0%, #ffffff 100%);
}

.submit-btn {
  background: linear-gradient(135deg, #6DC1F7 0%, #BCF1F2 100%);
}

/* 音频播放器样式优化 */
.audio-player {
  width: 300px; /* ✅ 或者 auto + min-width */
  min-width: 250px;
  border-radius: 6px;
  margin-top: 0;
}

.waveform {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 30px;
  margin-top: 10px;
}

.bar {
  width: 4px;
  background-color: #409eff;
  animation: bounce 1s infinite ease-in-out;
  border-radius: 2px;
}

.bar:nth-child(1) { animation-delay: 0s; }
.bar:nth-child(2) { animation-delay: 0.1s; }
.bar:nth-child(3) { animation-delay: 0.2s; }
.bar:nth-child(4) { animation-delay: 0.3s; }
.bar:nth-child(5) { animation-delay: 0.4s; }
.bar:nth-child(6) { animation-delay: 0.3s; }
.bar:nth-child(7) { animation-delay: 0.2s; }
.bar:nth-child(8) { animation-delay: 0.1s; }
.bar:nth-child(9) { animation-delay: 0s; }
.bar:nth-child(10){ animation-delay: 0.05s; }

@keyframes bounce {
  0%, 100% {
    height: 6px;
  }
  50% {
    height: 28px;
  }
}


</style>
