def termFrequencyInDoc(vocab, doc_dict):
  tf_docs = {}
  for doc_id in doc_dict.keys():
    tf_docs[doc_id] = {}
  for word in vocab:
    for doc_id,doc in doc_dict.items():
      tf_docs[doc_id][word] = doc.count(word)
  return tf_docs

def tokenisasi(text):
    tokens = text.split(" ")
    return tokens

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
factory = StemmerFactory()
stemming = factory.create_stemmer()

import os, re
from spacy.lang.id import Indonesian
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import os

path = "D:/RAIHAN STIS/Perkuliahan/SEMESTER 5/Praktikum INFORMATION RETRIEVAL/Pertemuan (2)/berita"


file_list = sorted(os.listdir(path))

nlp_tugas = Indonesian()
stp_words_tugas = nlp_tugas.Defaults.stop_words
berita = []

for file_name in file_list:
    file_path = os.path.join(path, file_name)

    with open(file_path, 'r') as f:
        clean_txt = re.sub("http\S+", ' ', f.read())
        clean_txt = re.sub("[^\w\s0-9]|['\d+']|[\'\",.!?:;<>()\[\]{}@#$%^&*=_+/\\\\|~-]]|(\'\')", ' ', clean_txt)
        clean_txt = re.sub("[\n\n]", ' ', clean_txt)
        clean_txt = re.sub(r'\s+', ' ', clean_txt).strip()
        berita.append(clean_txt)

doc_dict = {}
for i in range(1, len(berita) + 1):
    words = berita[i - 1].split()
    # eliminasi stopwords
    filtered_words = [word for word in words if word.lower() not in stp_words_tugas]
    # stemming
    stemmed_words = [stemming.stem(word) for word in filtered_words]

    # karena sebelumnya masih dalam per kata disatukan kembali menjadi kalimat untuk variabel doc_dict
    doc_dict[i] = " ".join(stemmed_words)
print(doc_dict)

token_arrays = []
for doc in berita:
    text_low = doc.lower()
    nlp_doc = nlp_tugas(text_low)
    token_doc = [token.text for token in nlp_doc]
    token_stpwords_tugas = [w for w in token_doc if w not in stp_words_tugas]
    token_arrays.append(token_stpwords_tugas)
print(token_arrays)

inverted_index = {}
for i in range(len(token_arrays)):
    for item in token_arrays[i]:
        item = stemming.stem(item)
        if item not in inverted_index:
            inverted_index[item] = []
        if (item in inverted_index) and ((i+1) not in inverted_index[item]):
            inverted_index[item].append(i+1)
print(inverted_index)

vocab = list(inverted_index.keys())


def wordDocFre(vocab, doc_dict):
  df = {}
  for word in vocab:
    frq = 0
    for doc in doc_dict.values():
      if word in tokenisasi(doc):
        frq = frq + 1
    df[word] = frq
  return df

import numpy as np
def inverseDocFre(vocab,doc_fre,length):
  idf= {}
  for word in vocab:
    idf[word] = idf[word] = 1 + np.log((length + 1) / (doc_fre[word]+1))
  return idf

def tfidf(vocab,tf,idf_scr,doc_dict):
  tf_idf_scr = {}
  for doc_id in doc_dict.keys():
    tf_idf_scr[doc_id] = {}
  for word in vocab:
    for doc_id,doc in doc_dict.items():
      tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
  return tf_idf_scr

tf_idf = tfidf(vocab, termFrequencyInDoc(vocab, doc_dict), inverseDocFre(vocab, wordDocFre(vocab, doc_dict), len(doc_dict)), doc_dict)

# Term - Document Matrix
TD = np.zeros((len(vocab), len(doc_dict)))
for word in vocab:
  for doc_id,doc in tf_idf.items():
    ind1 = vocab.index(word)
    ind2 = list(tf_idf.keys()).index(doc_id)
    TD[ind1][ind2] = tf_idf[doc_id][word]
print(TD)