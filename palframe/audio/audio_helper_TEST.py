#!/usr/bin/env python
#coding: utf8
#author: Tian Xia

from palframe.audio.audio_helper import AudioHelper
from palframe import common as nlp
import os

if __name__ == '__main__':
  data_path = os.path.join(nlp.get_module_path("common"), "audio/audio")

  files = nlp.get_files_in_folder(data_path,
                                  file_extensions=AudioHelper.AUDIO_EXTENSIONS)

  for in_file in files:
    # out_file = AudioHelper.convert_to_flac(in_file)
    out_file = AudioHelper.convert_to_wav(in_file)
    if out_file is not None:
      file_info = AudioHelper.get_basic_audio_info(out_file)
      length = file_info["duration"]
      length_str = nlp.to_readable_time(length)
      print(f"[OK] {in_file} to {out_file}, {length_str}")

    else:
      print(f"[ERR] {in_file}")
      assert False
