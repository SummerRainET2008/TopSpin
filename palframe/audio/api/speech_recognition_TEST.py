#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin

import palframe.audio.api.speech_recognition as SpeechRecognition
import optparse

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-d', '--device_index', default=1)
  parser.add_option('-p', '--phrase_time_limit', default=10)
  parser.add_option('-t', '--timeout', default=5)
  parser.add_option('-l', '--language', default='cmn-Hans-CN')
  parser.add_option('-a', '--audio_file', default='company')
  (options, args) = parser.parse_args()
  SpeechRecognition.listen_and_recognize(options.timeout, options.language)
  # for i in range(100):
  #   SpeechRecognition.audio_file_recognize(options.audio_file + str(i) +
  #                                          '.wav', options.language)
