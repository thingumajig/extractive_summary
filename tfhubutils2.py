#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

import tensorflow as tf

from functools import lru_cache


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


class TFHubContext2:
                         
  def __init__(self, url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3") -> None:
    super().__init__()
    print("initialize model.")

    self.model = hub.load(url)

  @lru_cache(maxsize=512)
  def get_embedding(self, texts):
    # Reduce logging output.
    # tf.logging.set_verbosity(tf.logging.ERROR)

    return self.model(texts)

  def close(self):
    print('TFHubContext closed')


from functools import lru_cache

@lru_cache(maxsize=10)
# def get_sentence_encoder(name='universal-sentence-encoder-multilingual-large/3'):
def get_sentence_encoder(name='universal-sentence-encoder-large/5'):
  return TFHubContext2(url=f'https://tfhub.dev/google/{name}')

