# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

from tensorflow import keras
import params_flow as pf

from palframe.tf_2x.bert.open_source.bert.layer import Layer
from palframe.tf_2x.bert.open_source.bert.embeddings import BertEmbeddingsLayer
from palframe.tf_2x.bert.open_source.bert.transformer import TransformerEncoderLayer


class BertModelLayer(Layer):
    """
    Implementation of BERT (arXiv:1810.04805), adapter-BERT (arXiv:1902.00751) and ALBERT (arXiv:1909.11942).

    See: https://arxiv.org/pdf/1810.04805.pdf - BERT
         https://arxiv.org/pdf/1902.00751.pdf - adapter-BERT
         https://arxiv.org/pdf/1909.11942.pdf - ALBERT

    """
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayer.Params):
        pass

    # noinspection PyUnusedLocal
    def _construct(self, params: Params):
        self.embeddings_layer = None
        self.encoders_layer   = None

        self.support_masking  = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )

        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayer.from_params(
            self.params,
            name="encoder"
        )

        super(BertModelLayer, self).build(input_shape)

    def apply_adapter_freeze(self):
        """ Should be called once the model has been built to freeze
        all bet the adapter and layer normalization layers in BERT.
        """
        if self.params.adapter_size is not None:
            def freeze_selector(layer):
                return layer.name not in ["adapter-up", "adapter-down", "LayerNorm", "extra_word_embeddings"]
            pf.utils.freeze_leaf_layers(self, freeze_selector)

    def call(self, inputs, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        output           = self.encoders_layer(embedding_output, mask=mask, training=training)
        return output   # [B, seq_len, hidden_size]

