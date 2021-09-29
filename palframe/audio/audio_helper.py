#!/usr/bin/env python
#coding: utf8
#author: Tian Xia 

# from mutagen.mp3 import MP3
# import audioread
from palframe.nlp import print_flush, execute_cmd
from pydub import AudioSegment
from pydub.utils import mediainfo
from palframe import nlp
from scipy.io import wavfile
import os
import typing
import numpy

class AudioHelper:
  AUDIO_EXTENSIONS = [
    "mp3",      # converted to wav
    "flac",     # target format
    "wav",
    "sph"       # converted to wav
  ]

  @staticmethod
  def segment_audio(flac_or_wav_file: str, time_segments: list,
                    dest_folder: str)-> typing.Iterator:
    '''
    time_segments: [(12,97, 18.89), (18.43, 27.77) ...] in seconds.
    return: an iterator retuning a new segment file. If one time segment is
    invalid, then the its corresponding segment file name is None.
    '''
    file_ext = nlp.get_file_extension(flac_or_wav_file)
    assert file_ext in ["flac", "wav"]
    assert os.path.exists(dest_folder)

    base_name = os.path.basename(flac_or_wav_file)
    audio = AudioSegment.from_file(flac_or_wav_file, file_ext)
    duration = len(audio)
    for file_id, (time_from, time_to) in enumerate(time_segments):
      t_from = time_from * 1000
      t_to = time_to * 1000

      if not (0 <= t_from < t_to < duration):
        print(f"WARN: {flac_or_wav_file} is not complete. "
              f"Actual length: {duration / 1000} seconds, "
              f"while time segment is {time_from}-{time_to}")
        yield None
        continue

      seg_name = os.path.join(
        dest_folder,
        nlp.replace_file_name(
          base_name, f".{file_ext}", f".{file_id:04}.{file_ext}"
        )
      )
      try:
        audio[t_from: t_to].export(seg_name, format=file_ext)
        yield seg_name

      except Exception as error:
        print_flush(error)
        yield None

  @staticmethod
  def get_detailed_audio_info(audio_file: str)-> dict:
    return mediainfo(audio_file)

  @staticmethod
  def get_basic_audio_info(audio_file: str)-> dict:
    file_ext = nlp.get_file_extension(audio_file)

    audio = AudioSegment.from_file(audio_file, file_ext)
    channels = audio.channels    #Get channels
    sample_width = audio.sample_width #Get sample width
    duration_in_sec = len(audio) / 1000 #Length of audio in sec
    sample_rate = audio.frame_rate
    bit_per_sample = sample_width * 8
    bit_rate = sample_rate * bit_per_sample * channels
    #in bytes.
    # file_size = (sample_rate * bit_rate * channel_count * duration_in_sec) / 8
    # print(f"audio file size: {file_size} bytes")

    return {
      "file": audio_file,
      "channels": channels,
      "sample_width": sample_width,
      "duration": duration_in_sec,
      "sample_rate": sample_rate,
      "bit_per_sample": bit_per_sample,
      "bit_rate": bit_rate,
    }

  @staticmethod
  def _convert_flac_to_wav(flac_file: str)-> typing.Union[str, None]:
    assert flac_file.endswith(".flac")
    out_file = nlp.replace_file_name(flac_file, ".flac", ".wav")
    if os.path.exists(out_file):
      return out_file

    cmd = f"sox {flac_file} {out_file}"
    if nlp.execute_cmd(cmd) == 0:
      return out_file

    return None

  @staticmethod
  def preemphasize_wav(standard_wav_file: str):
    assert standard_wav_file.endswith(".norm.wav")
    new_file = nlp.replace_file_name(
      standard_wav_file, ".norm.wav", ".norm.amp.wav"
    )
    if os.path.exists(new_file):
      return new_file

    sample_rate, signal = wavfile.read(standard_wav_file)
    assert sample_rate == 16000

    pre_emphasis = 0.97
    emphasized_signal = numpy.append(
      signal[0], signal[1:] - pre_emphasis * signal[:-1]
    )
    amplified_signal = emphasized_signal * (32768 / emphasized_signal.max())
    wavfile.write(new_file, sample_rate, amplified_signal.astype(numpy.int16))

    return new_file

  @staticmethod
  def convert_to_standard_wav(wav_or_flac_file: str)-> typing.Union[str, None]:
    '''
    :return: sample-width=2 Bytes, sample rating=16K.
    '''
    if wav_or_flac_file.endswith(".norm.wav"):
      return wav_or_flac_file

    file_ext = nlp.get_file_extension(wav_or_flac_file)
    new_file = nlp.replace_file_name(wav_or_flac_file,
                                     f".{file_ext}", ".norm.wav")
    if os.path.exists(new_file):
      return new_file

    if nlp.execute_cmd(f"sox {wav_or_flac_file} "
                       f"-b 16 -r 16000 {new_file}") == 0:
      return new_file

    return None

  @staticmethod
  def convert_to_16bits(wav_or_flac_file: str)-> typing.Union[str, None]:
    file_ext = nlp.get_file_extension(wav_or_flac_file)
    new_file = nlp.replace_file_name(wav_or_flac_file,
                                     f".{file_ext}", ".16bits.wav")
    if os.path.exists(new_file):
      return new_file

    if nlp.execute_cmd(f"sox {wav_or_flac_file} -b 16 {new_file}") == 0:
      return new_file

    return None

  @staticmethod
  def convert_to_wav(in_file: str)-> typing.Union[str, None]:
    file_ext = nlp.get_file_extension(in_file)

    if file_ext == "mp3":
      return AudioHelper._convert_mp3_to_wav(in_file)

    elif file_ext == "flac":
      return AudioHelper._convert_flac_to_wav(in_file)

    elif file_ext == "wav":
      return in_file

    elif file_ext == "sph":
      return AudioHelper._convert_sph_to_wav(in_file)

    else:
      assert False, \
        f"{in_file} extension is not in {AudioHelper.AUDIO_EXTENSIONS}"

  @staticmethod
  def convert_to_flac(in_file: str)-> typing.Union[str, None]:
    ''' Return in_file: return flac format.
    Only convert files appearing in AUDIO_EXTENSIONS'''
    file_ext = nlp.get_file_extension(in_file)

    if file_ext == "mp3":
      out_file = AudioHelper._convert_mp3_to_wav(in_file)
      if out_file is not None:
        return AudioHelper.convert_to_flac(out_file)
      return None

    elif file_ext == "flac":
      return in_file

    elif file_ext == "wav":
      out_file = AudioHelper._convert_wav_to_flac(in_file)
      if out_file is not None:
        return AudioHelper.convert_to_flac(out_file)
      return None

    elif file_ext == "sph":
      out_file = AudioHelper._convert_sph_to_wav(in_file)
      if out_file is not None:
        return AudioHelper.convert_to_flac(out_file)
      return None

    else:
      assert False, \
        f"{in_file} extension is not in {AudioHelper.AUDIO_EXTENSIONS}"

  @staticmethod
  def _convert_wav_to_flac(wav_file: str)-> typing.Union[str, None]:
    assert wav_file.endswith(".wav")
    out_file = nlp.replace_file_name(wav_file, ".wav", ".flac")
    if os.path.exists(out_file):
      return out_file

    cmd = f"sox {wav_file} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    return None

  @staticmethod
  def _convert_mp3_to_wav(map3_file: str)-> typing.Union[str, None]:
    assert map3_file.endswith(".mp3")
    out_file = nlp.replace_file_name(map3_file, ".mp3", ".wav")
    if os.path.exists(out_file):
      return out_file

    cmd = f"ffmpeg -i {map3_file} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    return None

  @staticmethod
  def _convert_sph_to_wav(sph_file: str)-> typing.Union[str, None]:
    assert sph_file.endswith(".sph")
    out_file = nlp.replace_file_name(sph_file, ".sph", ".wav")
    if os.path.exists(out_file):
      return out_file

    cmd = f"sox {sph_file} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    cmd = f"sph2pipe -f rif {sph_file} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    return None

