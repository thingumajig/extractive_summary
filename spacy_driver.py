#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import spacy
from spacy.symbols import *

import os
import numpy as np
from tfhubutils2 import TFHubContext2


class SpacyParser(object):
  __slots__ = ['model_name', 'nlp', 'disable']

  # print(dir(spacy.symbols))
  # subj - subject
  # nsubj - nominal subject
  # nsubjpass - passive nominal subject
  # csubj - clausal subject
  # csubjpass - passive clausal subject
  en_np_labels = {nsubj, nsubjpass, dobj, iobj, pobj}  # Probably others too
  en_np_labels_full = {nsubj, nsubjpass, dobj, iobj, pobj, csubj, csubjpass, attr}  # Probably others too


  def __init__(self, model_name='en_core_web_lg', disable=['ner']) -> None:
    super().__init__()

    self.model_name = model_name
    self.disable = disable

    self.__internal_load()

  def __internal_load(self):
    if self.model_name:
      os.system(f'python -m spacy download {self.model_name}')

    self.nlp = spacy.load(self.model_name)

  def parse(self, text):
    return self.nlp(text, disable=self.disable)

  def get_sentences(self, doc):
    sentences = []
    for sent in doc.sents:
      s = str(sent).strip()
      if s:
        sentences.append(s)
      else:
        print('Empty sentence!')

    # return [str(sent) for sent in doc.sents]
    return sentences

  def get_sentence_structs(self, doc):
    return [sent for sent in doc.sents]

  def iter_nps(self, doc):
      for word in doc:
        if word.dep in SpacyParser.en_np_labels_full:
          yield word

  def iter_nps_str(self, doc):
    for np in self.iter_nps(doc):
      s = ''
      for t in np.subtree:
        s += str(t) + ' '
      yield s.strip()




class SpacyDocExtra(object):
  __slots__ = ['sentences', 'doc', 'embeddings', 'parser', 'embedder', 'index', 'sentence_structs', 'text']

  def __init__(self, text, parser=SpacyParser(), embedder = TFHubContext2()) -> None:
    super().__init__()
    self.parser = parser
    self.embedder = embedder
    self.text = text

    self.doc = self.parser.parse(text)

    # self.sentences = self.parser.get_sentence_structs(self.doc)
    self.sentences = []
    self.sentence_structs = []
    for sent in self.doc.sents:
      s = str(sent).strip()
      if s:
        self.sentences.append(s)
        self.sentence_structs.append(sent)
      else:
        print('Empty sentence!')



    self.embeddings = self.embedder.get_embedding(self.sentences)
    self.embeddings = np.array(self.embeddings).tolist()

    self.index = dict([(v, i) for i, v in zip(range(len(self.sentences)), self.sentences)])


  def get_emb(self, text):
    return self.embeddings[self.index[text]]



if __name__ == '__main__':
  s = SpacyParser(model_name='en_core_web_md')
  text = '''
  The House Judiciary Committee's vote Friday passing articles of impeachment against President Donald Trump was meant as extraordinary repudiation.
The history was certainly there, and is weighing on the President. But it did not appear anyone felt rebuked at the White House.
  '''

  doc = s.parse(text);
  for sent in doc.sents:
    print(sent)