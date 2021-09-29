#coding: utf8
#author: Tian Xia 

import optparse
import os
from palframe.audio.acoustic_feature import *
from palframe.audio.audio_helper import AudioHelper

if __name__ == "__main__":
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  (options, args) = parser.parse_args()

  audio_file = os.path.join(
    nlp.get_module_path("palframe/common"),
    "palframe/audio/audio/102-129232-0009.flac"
  )

  start = time.time()
  audio_info = AudioHelper.get_basic_audio_info(audio_file)
  print(audio_info)
  mfcc_dim = 100
  features = calc_mfcc_delta(audio_file, mfcc_dim)
  print(f"#frame: {len(features)}, #feat: {len(features[0])}")
  print(f"duration: {time.time() - start}")

  parallel_calc_features(
    [audio_file] * 1000, mfcc_dim, "feature", 8
  )

