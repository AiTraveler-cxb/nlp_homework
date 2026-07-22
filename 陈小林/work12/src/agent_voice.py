import time

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from readerwriterlock.rwlock import RWLockWrite
import AppKit

from src.memory.base import ConversationMemory
from src.llm.base import LLMFactory
from src.models.entities import LLMConfig
import pyttsx3
import torch
import numpy as np
import pyaudio
from silero_vad import VADIterator, load_silero_vad
from faster_whisper import WhisperModel
import collections
import threading
import queue

from faster_whisper import WhisperModel



class Agent(BaseModel):

    memories:dict[str,ConversationMemory] = Field(default_factory=dict)
    llm_config:LLMConfig = None
    llm:ChatOpenAI = None

    def model_post_init(self, _context):
        self.llm = LLMFactory.create_llm(self.llm_config)

    def chat(self, session_id:str, message:str):
        memory = self.memories.get(session_id)
        if memory is None:
            memory = ConversationMemory(session_id=session_id)
            self.memories[session_id] = memory
        history = memory.get_context(message)

        res = self.llm.invoke(history)
        memory.add_message(message, res.content)
        return res.content


class VoiceAgent:

    def __init__(self,
                 agent: Agent,
                 sample_rate: int = 16000,
                 chunk_size: int = 512
                 ):
        self.vad_model = load_silero_vad()
        self.sample_rate = sample_rate
        self.agent = agent
        self.vad_iterator = VADIterator(model=self.vad_model, sampling_rate = sample_rate)
        self.tts = pyttsx3.init()
        #模型
        self.model = WhisperModel("base", device="cpu", compute_type="int8")

        # 音频参数
        self.chunk_size = chunk_size  # VAD 需要的 chunk size
        self.audio_buffer = collections.deque(maxlen=self.sample_rate * 30)  # 保留30秒

        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

        # 队列
        self.text_queue = queue.Queue()

        #互斥
        self.lock = threading.Lock()
        self.ttsing = False
        self.rw_lock = RWLockWrite()
    def process_audio(self, audio_chunk):
        """处理音频流"""
        # 转换为 tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # VAD 检测是否有语音
        speech_dict = self.vad_iterator(audio_tensor)

        if speech_dict:
            if 'start' in speech_dict:
                print("🎤 检测到语音开始...")
                self.audio_buffer.clear()
            elif 'end' in speech_dict:
                print("🎤 语音结束，开始识别...")
                # 获取音频片段
                audio_segment = np.array(list(self.audio_buffer))

                # Whisper 识别
                segments, _ = self.model.transcribe(
                    audio_segment,
                    language="zh",
                    beam_size=5
                )
                text = "".join([seg.text for seg in segments])

                if text:
                    print(f"📝 {text}")
                    self.text_queue.put(text)

        # 缓存音频
        self.audio_buffer.extend(audio_chunk)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """实时回调"""
        audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        threading.Thread(target=self.process_audio, args=(audio_chunk,)).start()
        return (None, pyaudio.paContinue)

    def run(self, session_id: str):
        """主循环"""
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )

        print("🎙️ 实时语音助手启动（说话会自动识别）")
        self.stream.start_stream()

        try:
            while True:
                if not self.text_queue.empty():
                    user_input = self.text_queue.get()
                    response = self.agent.chat(session_id, user_input)
                    self.stream.stop_stream()
                    self.stream.close()
                    self.say_sync(response)
                    self.text_queue = queue.Queue()
                    self._init_stream()
        except KeyboardInterrupt as e:
            print('语音输出异常。。。。。。。。。。。。。。。', e)
            self.stream.stop_stream()
        finally:
            self.cleanup()

    def say_sync(self, response):
        """使用 AppKit 的 NSSpeechSynthesizer"""
        synth = AppKit.NSSpeechSynthesizer.alloc().init()
        synth.startSpeakingString_(response)

        # 等待播放完成
        while synth.isSpeaking():
            AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )

        return True

    def _init_stream(self):
        """创建并启动音频流 (用于替代原本的 start_stream 逻辑)"""
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
