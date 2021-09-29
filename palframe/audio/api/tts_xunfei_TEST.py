#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin

import palframe.audio.api.tts_xunfei as TextToSpeech

if __name__ == '__main__':
  TextToSpeech.text_to_speech('苟利国家生死以，岂因祸福避趋之', 'audio001')
