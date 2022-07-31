from example.nlp.intent_detection import *
from example.nlp.intent_detection.param import Param


class Model(nn.Module):
  def __init__(self, param: Param):
    super(Model, self).__init__()

    self._embedding = nn.Embedding(param.vocab_size, param.embedding_size)

    self._textcnn = nlp_torch.TextCNN(kernels=param.kernel_sizes,
                                      in_channel=1,
                                      out_channel=param.kernel_number,
                                      max_seq_len=param.max_seq_len,
                                      dim=param.embedding_size,
                                      dropout=param.dropout_ratio)
    self._textcnn_output_size = len(param.kernel_sizes) * param.kernel_number

    self._dense = nlp_torch.Dense(
        nn.Linear(self._textcnn_output_size, param.class_number))

  def forward(self, word_ids):
    word_ids = word_ids.unsqueeze(1)
    embedding_out = self._embedding(word_ids)
    textcnn_out = self._textcnn(embedding_out)
    out = self._dense(textcnn_out)
    pred_labels = torch.argmax(out, 1)

    return out, pred_labels
