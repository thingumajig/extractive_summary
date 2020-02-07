#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from collections import defaultdict
import numpy as np
from functools import lru_cache

from sklearn.cluster import Birch
from scipy.spatial import distance

from spacy_driver import SpacyParser, SpacyDocExtra
from tfhubutils2 import *


class ExtractiveSummary(object):
  __slots__ = ['parser', 'embedder', 'extradoc', 'text', 'centroids', 'name_candidates_dict']

  def __init__(self, parser=SpacyParser(), embedder = TFHubContext2()) -> None:
    super().__init__()

    self.parser = parser
    self.embedder = embedder
    self.text = ''
    self.name_candidates_dict = None

  def preprocess_text(self, t):
    t = t.strip()
    if self.text != t:
      self.name_candidates_dict = None
      self.centroids = None
      self.extradoc = SpacyDocExtra(t, parser=self.parser, embedder=self.embedder)



  def cluster_embeddings(self, threshold=0.4, min_cluster_elements=3, n_clusters=0):
    if n_clusters == 0:
      n_clusters = None

    bm = Birch(threshold=threshold, n_clusters=n_clusters)
    # bm = Birch(n_clusters=5)
    bm.fit(self.extradoc.embeddings)

    labels = bm.labels_
    print(f'labels: {labels} ({len(labels)})')
    self.centroids = bm.subcluster_centers_
    print(f'centroids:  {len(self.centroids)} ')
    n_clusters = np.unique(labels).size

    print(f'n_clusters:{n_clusters}')

    clusters = defaultdict(list)
    for i, key in enumerate(labels):
      clusters[key].append(i)

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4, width=60, compact=True)
    # pp.pprint(clusters)

    # selected_clusters = []
    # for k, v in clusters.items():
    #   if len(v) >= min_cluster_elements:
    #     selected_clusters.append((k, v))

    selected_clusters = list(filter(lambda elem: len(elem[1]) >= min_cluster_elements, clusters.items()))
    selected_clusters = list(sorted(selected_clusters, key=lambda elem: len(elem[1]),reverse=True))

    if n_clusters:
      selected_clusters = selected_clusters[:n_clusters]


    # pp.pprint(selected_clusters)

    # print('Themes:')
    # for x in selected_clusters:
    #   print(f'N{x[0]}{"-" * 30}')
    #   for sent in x[1]:
    #     print(es.sentences[sent])
    return selected_clusters

  # @lru_cache(maxsize=15)
  def get_extractive_texts(self, t, threshold=0.7, min_clusters_elements=3, n_clusters=0, minimum_sentence_len=20, max_naming_len = 30, distance_threshold = 0.95):
    self.preprocess_text(t)
    selected_clusters = self.cluster_embeddings(threshold, min_cluster_elements=min_clusters_elements, n_clusters=n_clusters)

    self.name_candidates_dict = self.collect_keyphrases(self.extradoc.doc, max_naming_len = max_naming_len)

    selected_texts = []
    for s in selected_clusters:
      cluster_ix = s[0]
      centroid = self.centroids[cluster_ix]

      #Filter sentences
      selected_sentences = []
      name_candidates = set()
      for ns in s[1]:
        sent = self.extradoc.sentences[ns]
        # sent = str(sent).strip()
        if len(sent) > minimum_sentence_len:
          selected_sentences.append(sent)

        #Collect key phrases candidates
        ssent = self.extradoc.sentence_structs[ns]
        for spart in self.parser.iter_nps_str(ssent):
          if len(spart) <= max_naming_len:
            name_candidates.add(spart)

      if len(selected_sentences) >= min_clusters_elements:
        name_candidates_list = []
        for name in name_candidates:
          emb = self.name_candidates_dict[name]
          d = distance.cosine(emb, centroid)
          if d<=distance_threshold:
            name_candidates_list.append((name, d))

        name_candidates_list = sorted(name_candidates_list, key=lambda x: x[1])

        selected_texts.append((cluster_ix, selected_sentences, name_candidates_list))

    return selected_texts

  def collect_keyphrases(self, doc, max_naming_len = 30):
    name_candidates = set()
    for spart in self.parser.iter_nps_str(doc):
      if len(spart) <= max_naming_len:
        name_candidates.add(spart)


    name_candidates = list(name_candidates)
    name_embeddings = self.embedder.get_embedding(name_candidates)

    name_candidates_dict = defaultdict()
    for i, name in enumerate(name_candidates):
      name_candidates_dict[name] = name_embeddings[i]


    return name_candidates_dict

  def create_word_cloud(self, selected_texts):
    wcdict = defaultdict()

    X = []
    y = []
    for e in selected_texts:
      for r in e[2]:
        # wcdict[r[0]]=1-r[1]
        X.append([1 - r[1]])
        y.append(r[0])

    from sklearn.preprocessing import MinMaxScaler
    scaler:MinMaxScaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(X)
    for i, v in enumerate(X):
      wcdict[y[i]] = X[i][0]

    # pp.pprint(wcdict)

    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    wordcloud = WordCloud(background_color="white", width=800, height=600,
                          min_font_size=6, font_step=2)
    wordcloud.generate_from_frequencies(wcdict)
    return wordcloud


def create_extractive_summary_gen(model_name='en_core_web_sm', emb_name='universal-sentence-encoder-multilingual-large/3'):
  parser = SpacyParser(model_name=model_name)
  embedder = get_sentence_encoder(name=emb_name)
  return ExtractiveSummary(parser=parser, embedder=embedder)


if __name__ == '__main__':
  with open('test_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

    es = ExtractiveSummary()

    es.preprocess_text(text)
    es.cluster_embeddings()
