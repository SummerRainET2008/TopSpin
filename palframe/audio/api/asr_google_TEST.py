#coding: utf8
#author: Xinlu Yu

from palframe.audio.api.asr_google import audio_recognition
from palframe import common as common
import os
'''
Using googleApi key
Chinese please use language="Zh-cn"
English please use language="US-en"
'''
if __name__ == '__main__':
  data_path = os.path.join(
    common.get_module_path("common"),
    "audio/audio.wav"
  )
  audio_recognition(data_path, language="Zh-cn")
