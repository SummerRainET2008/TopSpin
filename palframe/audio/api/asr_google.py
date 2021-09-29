#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu

import os
from palframe import nlp as common
import io
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

'''
Google audio Api:支持的语音格式为FLAC或WAV或SPEEX
'''
document_path = common.get_module_path("audio.SpeechRecognition")
credential_path = os.path.join(document_path,
                               "audio/paii-nlp-service-2eea7bc1f421.json")
print(credential_path)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
def audio_recognition(file_name, language='en-US'):
  client = speech.SpeechClient()
  # The name of the audio file to transcribe
  # Loads the audio into memory
  with io.open(file_name, 'rb') as audio_file:
    content = audio_file.read()
    audio = types.RecognitionAudio(content=content)

  config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code=language)

  # Detects audio in the audio file
  response = client.recognize(config, audio)

  for result in response.results:
    print('Transcript: {}'.format(result.alternatives[0].transcript))