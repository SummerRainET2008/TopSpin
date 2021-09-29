#coding: utf8
#author: Tian Xia 

from palframe.tf_1x.nlp_tf import *
from palframe.tf_1x.crf.lstm_crf.data import *
from palframe.tf_1x.crf.lstm_crf.trainer import Trainer

class Predictor(object):
  def __init__(self, model_path):
    ''' We only load the best model in {model_path}
    '''
    def extract_id(model_file):
      return int(re.findall(r"iter-(.*?).index", model_file)[0])

    param_file = os.path.join(model_path, "param.pydict")
    self.param = list(read_pydict_file(param_file))[0]
    
    self._vob = Vocabulary(self.param["remove_OOV"],
                           self.param["max_seq_length"])
    self._vob.load_model(self.param["vob_file"])

    names = [extract_id(name) for name in os.listdir(model_path)
             if name.endswith(".index")]
    best_iter = max(names)
    model_prefix = f"{model_path}/iter-{best_iter}"
    print(f"loading model: '{model_prefix}'")
    
    graph = tf.Graph()
    with graph.as_default():
      saver = tf.train.import_meta_graph(f"{model_prefix}.meta")

    self._sess = tf.Session(graph=graph)
    with self._sess.as_default():
      with graph.as_default():
        saver.restore(self._sess, f"{model_prefix}")
    
  def translate(self, word_list: list, predict_seq: list):
    return Trainer.translate(word_list, predict_seq, self.param["tag_list"])

  def predict_dataset(self, file_name):
    data = DataSet(data_file=file_name,
                   tag_list=self.param["tag_list"],
                   vob=self._vob)
    data_iter = data.create_batch_iter(batch_size=self.param["batch_size"],
                                       epoch_num=1,
                                       shuffle=False)
    fou = open(file_name.replace(".pydict", ".pred.pydict"), "w")
    correct = 0.
    for batch_x, batch_y in data_iter:
      preds, probs, accuracy = self.predict(batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
      for idx, y in enumerate(batch_y):
        pred = {
          "predicted_tags": list(preds[idx]),
          "prob": probs[idx],
        }
        print(pred, file=fou)
    fou.close()
    
    accuracy = correct / data.size()
    print(f"Test: '{file_name}': {accuracy:.4f}")
    
  def predict_one_sample(self, word_list: list):
    '''
    :param word_list: must be processed by normalization.
    :return: [translation, prob]
    '''
    word_ids = self._vob.convert_to_word_ids(word_list)
    seq, prob, _ = self.predict([word_ids], None)
    seq, prob = seq[0], prob[0]
    
    tran = self.translate(word_list, seq)
    
    return tran, prob
    
  def predict(self, batch_x: list, batch_y: list):
    '''
    :param batch_x: must be of the length used in training.
    :param batch_y: could be None
    :return: [seq, seq_prob, accuracy]
    '''
    return Trainer.predict(self._sess, batch_x, batch_y)

