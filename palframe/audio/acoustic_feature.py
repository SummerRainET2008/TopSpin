#coding: utf8
#author: Tian Xia 

import librosa
import numpy as np
from palframe import nlp
import time
import multiprocessing as mp
from palframe.audio.audio_helper import AudioHelper

# output length = (seconds) * (sample rate) / (hop_length)
HOP_LENGTH = 160

def calc_mfcc_delta(wav_file_16bits: str, mfcc_dim: int,
                    file_sample_rate: int):
  '''
  :return: [mfcc, mfcc_delta, mfcc_delta2]
  '''
  assert wav_file_16bits.endswith(".wav")
  file_info = AudioHelper.get_basic_audio_info(wav_file_16bits)
  sample_width = file_info["sample_width"]
  assert sample_width == 2

  #librosa.load has be bug for sample_width=1
  #import scipy.io.wavfile as wavfile #this is correct
  #sample_rate, data = wavfile.read(wav_file_8_or_16_bits)
  #sr=None, means loading native sample rate; otherwise, resample to it.
  wav_data, real_sample_rate = librosa.load(wav_file_16bits, sr=None)
  assert real_sample_rate == file_sample_rate
  mfcc = np.transpose(
    librosa.feature.mfcc(
      wav_data, real_sample_rate, n_mfcc=mfcc_dim, hop_length=HOP_LENGTH
    ),
    [1, 0]
  )

  delta1 = librosa.feature.delta(mfcc)
  delta2 = librosa.feature.delta(mfcc, order=2)

  feature = []
  for v1, v2, v3 in zip(mfcc.tolist(), delta1.tolist(), delta2.tolist()):
    feature.append(v1 + v2 + v3)

  return feature

def parallel_calc_features(wav_files_16bits: list, mfcc_dim: int,
                           target_sample_rate: int, output_folder: str,
                           process_num: int, queue_capacity: int=1024):
  def run_process(process_id: int, audio_file_pipe: mp.Queue):
    count = 0
    with open(f"{output_folder}/part.{process_id}.feat", "w") as fou:
      while True:
        audio_file = audio_file_pipe.get()
        if audio_file is None:
          print(f"run_process[{process_id}] exits!")
          break

        feature = calc_mfcc_delta(audio_file, mfcc_dim, target_sample_rate)
        print([audio_file, feature], file=fou)

        count += 1
        if count % 100 == 0:
          nlp.print_flush(f"So far, process[{process_id}] have processed "
                          f"{count} audio files.")

  assert 1 <= process_num
  nlp.execute_cmd(f"rm -r {output_folder}")
  nlp.mkdir(output_folder)

  start_time = time.time()
  file_pipe = mp.Queue(queue_capacity)
  process_runners = [mp.Process(target=run_process, args=(idx, file_pipe))
                     for idx in range(process_num)]

  for p in process_runners:
    p.start()

  for audio_file in wav_files_16bits:
    file_pipe.put(audio_file)

  for _ in process_runners:
    file_pipe.put(None)

  for p in process_runners:
    p.join()

  duration = nlp.to_readable_time(time.time() - start_time)
  print(f"It takes {duration} to process {len(wav_files_16bits)} audio files")
