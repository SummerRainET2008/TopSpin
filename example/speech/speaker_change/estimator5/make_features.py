#coding: utf8
#author: Xinyi Wu

from example.speech.speaker_change.estimator5.estimator5 import *


def process_xvector_data(scp_file, output_path, sc_loc):
  data_paths = []
  with open(scp_file, "r") as reader:
    for line in reader:
      file_path = line.strip().split(" ")[1]
      data_paths.append(file_path)
  Logger.info(f"#files: {len(data_paths)}")
  data = []
  for file in data_paths:
    xvecs, spk_ids = [], []
    with open(file, "r") as reader:
      for d in reader:
        d = eval(d)
        spk_ids.append(d.get("spk_id", "0"))
        xvecs.append(d.get("xvector"))
    if len(spk_ids) > 1:
      label = int(spk_ids[-sc_loc] != spk_ids[-(sc_loc + 1)])
    else:
      label = 0
    data.append((xvecs, label))
    if len(data) > 0 and len(data) % 10000 == 0:
      Logger.info(f"{len(data)} files have been processed.")
  Logger.info(f"#data: {len(data)}")
  with open(output_path, 'wb') as f_out:
    pickle.dump(data, f_out)


if __name__ == '__main__':
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()
  Logger.set_level(options.debug_level)

  process_xvector_data('example/speech/speaker_change/data/train.scp',
                       'example/speech/speaker_change/feat/train.pkl', 6)
  process_xvector_data('example/speech/speaker_change/data/vali.scp',
                       'example/speech/speaker_change/feat/vali.pkl', 6)
  process_xvector_data('example/speech/speaker_change/data/test.scp',
                       'example/speech/speaker_change/feat/test.pkl', 6)
