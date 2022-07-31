#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin

# pip install SpeechRecognition
import speech_recognition as sr


def audio_file_recognize(audio_file, language_selection):
  '''
  :param audio_file: audio file to be recognized
  :param language_selection: selected language code
  :return: print recognized audio content
  :支持的文件格式:SpeechRecognition 目前支持的文件类型有：
    WAV: 必须是 PCM/LPCM 格式, AIFF, AIFF-C
    FLAC: 必须是初始 FLAC 格式；OGG-FLAC 格式不可用
  '''
  r = sr.Recognizer()
  stock_audio = sr.AudioFile(audio_file)
  with stock_audio as source:
    audio = r.record(source)
    print(r.recognize_google(audio, language=language_selection))


def listen_and_recognize(time_out, language_selection):
  '''
  :param input_device_index: assigned index for the specific input device
  :param time_limit: set min time for the listening time
  :param time_out: set max time for the listening time
  :param language_selection: selected language code
  :return: print recognized audio content
  '''
  r = sr.Recognizer()
  mic = sr.Microphone()
  with mic as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, timeout=time_out)
  try:
    print(r.recognize_google(audio, show_all=True,
                             language=language_selection))
  except:
    print("Sorry, I cannot understand your questions.")
