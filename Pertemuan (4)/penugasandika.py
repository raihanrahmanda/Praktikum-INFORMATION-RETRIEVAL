import os
from spacy.lang.id import Indonesian
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from spacy.lang.id.stop_words import STOP_WORDS
import math

nlp = Indonesian()

path = "D:/RAIHAN STIS/Perkuliahan/SEMESTER 5/Praktikum INFORMATION RETRIEVAL/Pertemuan (2)/berita"

def read_text_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content


def tokenisasi(text):
    tokens = text.split(" ")
    return tokens


# create stemmer
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed = stemmer.stem(text)
    return stemmed


def preprocess_text(text):
    # stemming
    stemmer = StemmerFactory().create_stemmer()
    stemmed_text = stemmer.stem(text)
    doc = nlp(stemmed_text)
    # tokenisasi
    tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    return tokens


inverted_index = {}

for idx, file in enumerate(os.listdir(path)):
    if os.path.isfile(os.path.join(path, file)) and file.endswith(".txt"):
        file_path = os.path.join(path, file)

        text = read_text_file(file_path)

        cleaned_tokens = preprocess_text(text)

        # Update inverted index
        for token in set(cleaned_tokens):  # menghindari duplikasi dokumen
            if token in inverted_index:
                inverted_index[token].append(idx + 1)
            else:
                inverted_index[token] = [idx + 1]

## 1. VECTOR SPACE MODEL

vocab = list(inverted_index.keys())
doc_dict = {}

## pengulangan untuk stemming seluruh file txt dalam folder berita
for idx, file in enumerate(os.listdir(path)):
    if os.path.isfile(os.path.join(path, file)) and file.endswith(".txt"):
        file_path = os.path.join(path, file)

        text = read_text_file(file_path)
        stemmed_text = stemming(text)
        doc_id = f"doc{idx + 1}"

        doc_dict[doc_id] = stemmed_text


def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}
    for word in vocab:
        for doc_id, doc in doc_dict.items():
            tf_docs[doc_id][word] = doc.count(word)
    return tf_docs

#print(termFrequencyInDoc(vocab, doc_dict))
import numpy as np


def wordDocFre(vocab, doc_dict):
    df = {}
    for word in vocab:
        frq = 0
        for doc in doc_dict.values():
            if word in tokenisasi(doc):
                frq = frq + 1
        df[word] = frq
    return df


def inverseDocFre(vocab, doc_fre, length):
    idf = {}
    for word in vocab:
        idf[word] = idf[word] = 1 + np.log((length + 1) / (doc_fre[word] + 1))
    return idf


def tfidf(vocab, tf, idf_scr, doc_dict):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id, doc in doc_dict.items():
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr


tf_idf = tfidf(
    vocab,
    termFrequencyInDoc(vocab, doc_dict),
    inverseDocFre(vocab, wordDocFre(vocab, doc_dict), len(doc_dict)),
    doc_dict,
)
# Term - Document Matrix
TD = np.zeros((len(vocab), len(doc_dict)))
for word in vocab:
    for doc_id, doc in tf_idf.items():
        ind1 = vocab.index(word)
        ind2 = list(tf_idf.keys()).index(doc_id)
        TD[ind1][ind2] = tf_idf[doc_id][word]
print("TD matrix dengan skor = tf_idf\n", TD)


## 2. UKURAN KEMIRIPAN
def edit_distance(string1, string2):
    if len(string1) > len(string2):
        difference = len(string1) - len(string2)
        string1[:difference]
        n = len(string2)
    elif len(string2) > len(string1):
        difference = len(string2) - len(string1)
        string2[:difference]
        n = len(string1)
    for i in range(n):
        if string1[i] != string2[i]:
            difference += 1
    return difference


def jaccard_sim(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# Euclidian Distance
def euclidian_dist(vec1, vec2):
    # subtracting vector
    temp = vec1 - vec2
    # doing dot product
    # for finding
    # sum of the squares
    sum_sq = np.dot(temp.T, temp)
    # Doing squareroot and
    # printing Euclidean distance
    return np.sqrt(sum_sq)


def cosine_sim(vec1, vec2):
    vec1 = list(vec1)
    vec2 = list(vec2)
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
        mag_1 = math.sqrt(sum([x**2 for x in vec1]))
        mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (mag_1 * mag_2)


from itertools import combinations

# Mendapatkan semua kombinasi pasangan dokumen
doc_combinations = list(combinations(doc_dict.keys(), 2))

# Mengiterasi melalui semua kombinasi pasangan dokumen
# for doc_pair in doc_combinations:
#     doc1_id, doc2_id = doc_pair
#     doc1_text = doc_dict[doc1_id]
#     doc2_text = doc_dict[doc2_id]

#     print(f"{doc1_id} dengan {doc2_id}")
#     print("Edit distance =", edit_distance(doc1_text, doc2_text))
#     print(
#         "Jaccard simmilarity =",
#         jaccard_sim(doc1_text.split(" "), doc2_text.split(" ")),
#     )
#     print(
#         "Euclidian distance =",
#         euclidian_dist(
#             TD[:, list(doc_dict.keys()).index(doc1_id)],
#             TD[:, list(doc_dict.keys()).index(doc2_id)],
#         ),
#     )
#     print(
#         "Cosine distance =",
#         cosine_sim(
#             TD[:, list(doc_dict.keys()).index(doc1_id)],
#             TD[:, list(doc_dict.keys()).index(doc2_id)],
#         ),
#     )
#     print("\n")
