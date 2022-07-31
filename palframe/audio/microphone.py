#coding: utf8
#author: Tian Xia

import wave
from pyaudio import PyAudio, paInt16
import time


class Microphone:
  SAMPLE_RATE = 44100
  SAMPLE_NUM = 8000
  CHANNEL_NUM = 1
  SAMPLE_WIDTH = 2
  MAX_TIME = 1

  def _save_wave_file(self, export_audio_name: str, data: list):
    with wave.open(export_audio_name, 'wb') as wf:
      wf.setnchannels(self.CHANNEL_NUM)
      wf.setsampwidth(self.SAMPLE_WIDTH)
      wf.setframerate(self.SAMPLE_RATE)
      wf.writeframes(b"".join(data))

    print(f"{export_audio_name} is saved!")

  def record(self, audio_name: str, max_time: int):
    input(f"press any key to begin to record {max_time} seconds voice >> ")

    stream = PyAudio().open(format=paInt16,
                            channels=self.CHANNEL_NUM,
                            rate=self.SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=self.SAMPLE_NUM)
    my_buf = []
    time_start = time.time()
    last_second = 0
    print(f"time: {last_second} s")
    while True:
      duration = time.time() - time_start
      if duration >= max_time:
        break
      if int(duration) != last_second:
        last_second = int(duration)
        print(f"time: {last_second} s")

      string_audio_data = stream.read(self.SAMPLE_NUM)
      my_buf.append(string_audio_data)

    stream.close()
    self._save_wave_file(audio_name, my_buf)

  def play(self, audio_file: str):
    wf = wave.open(audio_file, 'rb')
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    while True:
      data = wf.readframes(self.SAMPLE_NUM)
      if len(data) == 0:
        break
      stream.write(data)

    stream.close()
    p.terminate()
