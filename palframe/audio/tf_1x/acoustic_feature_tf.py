#coding: utf8
#author: Tian Xia 

import tensorflow as tf
import numpy
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import librosa
from palframe import common as nlp
from palframe.audio.audio_helper import AudioHelper

class DataGraphMFCC:
  # do NOT modify these numbers.
  window_duration       = 25        # ms
  stride_duration       = 10        # ms
  frame_num_per_second  = 100
  max_mfcc_num          = 40

  def __init__(self, sample_rate: int, dct_coef_count: int=-1):
    '''
    suppose the channel number is 1.
    '''
    assert sample_rate == 16_000
    if dct_coef_count == -1:
      dct_coef_count = DataGraphMFCC.max_mfcc_num
    else:
      assert dct_coef_count <= DataGraphMFCC.max_mfcc_num

    self._sample_rate = sample_rate
    samples_per_second = sample_rate / 1000
    window = int(DataGraphMFCC.window_duration * samples_per_second)
    stride = int(DataGraphMFCC.stride_duration * samples_per_second)

    self._graph = tf.Graph()
    with self._graph.as_default():
      self._in_wav_file = tf.placeholder(tf.string, [], name='wav_filename')
      self._in_frame_num = tf.placeholder(tf.int32, [])
      wav_loader = io_ops.read_file(self._in_wav_file)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
      self._out_audio = tf.squeeze(wav_decoder.audio)
      self._out_sample_rate = wav_decoder.sample_rate

      self._in_audio = tf.placeholder(tf.float32, [None])
      in_audio = tf.expand_dims(self._in_audio, -1)

      audio_clamp = tf.clip_by_value(in_audio, -1.0, 1.0)
      spectrogram = contrib_audio.audio_spectrogram(
        audio_clamp,
        window_size=window,
        stride=stride,
        magnitude_squared=True)
      self._out_spectrogram = spectrogram

      feat_ts = contrib_audio.mfcc(
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        dct_coefficient_count=dct_coef_count,
      )
      self._out_mfcc = feat_ts[0]
      self._out_real_mfcc_len = tf.shape(self._out_mfcc)[0]

      diff = tf.maximum(0, self._in_frame_num - self._out_real_mfcc_len)
      self._out_expanded_mfcc = tf.pad(
        self._out_mfcc,
        [[0, diff], [0, 0]],
      )[: self._in_frame_num]

    self._sess = tf.Session(graph=self._graph)
    print(f"DataGgraphMFCC graph is created!")

  def read_16bits_wav_file(self, wav_file: str):
    audio, sr = self._sess.run(
      fetches=[self._out_audio, self._out_sample_rate],
      feed_dict={
        self._in_wav_file: wav_file,
      }
    )
    assert sr == self._sample_rate

    return audio

  def calc_feats(self, audio_data: list, target_frame_num: int=-1):
    '''
    :return: [frame-num, feature, 3]
    '''
    if target_frame_num <= 0:
      mfcc, real_length = self._sess.run(
        fetches=[
          self._out_mfcc,
          self._out_real_mfcc_len,
        ],
        feed_dict={
          self._in_audio: audio_data,
        }
      )
      target_frame_num = real_length

    else:
      mfcc, real_length = self._sess.run(
        fetches=[self._out_expanded_mfcc, self._out_real_mfcc_len],
        feed_dict={
          self._in_audio: audio_data,
          self._in_frame_num: target_frame_num
        }
      )

    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta1)

    mfcc = mfcc.reshape([target_frame_num, -1, 1])
    delta1 = delta1.reshape([target_frame_num, -1, 1])
    delta2 = delta2.reshape([target_frame_num, -1, 1])
    feat = numpy.concatenate([mfcc, delta1, delta2], axis=2)

    return feat, real_length
