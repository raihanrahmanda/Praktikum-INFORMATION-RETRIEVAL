import os
from spacy.lang.id import Indonesian
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from spacy.lang.id.stop_words import STOP_WORDS

nlp = Indonesian()

path = "D:/RAIHAN STIS/Perkuliahan/SEMESTER 5/Praktikum INFORMATION RETRIEVAL/Pertemuan (2)/berita"

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def preprocess_text(text):
    stemmer = StemmerFactory().create_stemmer()
    stemmed_text = stemmer.stem(text)
    
    doc = nlp(stemmed_text)
    tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    
    return tokens

inverted_index = {}
doc_dict = {}
document_index = 1

for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)) and file.endswith(".txt"):
        file_path = os.path.join(path, file)

        text = read_text_file(file_path)

        cleaned_tokens = preprocess_text(text)
        doc_dict[document_index] = " ".join(cleaned_tokens)
        document_index += 1

        # Update the inverted index
        for token in set(cleaned_tokens):  # Use set to avoid duplicate documents
            inverted_index.setdefault(token, []).append(file)
        
vocab=list(inverted_index.keys())

def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_docs[doc_id][word] = doc.count(word)
    return tf_docs

def wordDocFre(vocab, doc_dict):
    df = {}
    for word in vocab:
        frq = 0
        for doc in doc_dict.values():
            if word in doc:
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

print("\nSkor Kemiripan dengan Edit Distance")
print("Berita 1 dengan berita 2 : ", edit_distance(doc_dict[1], doc_dict[2]))
print("Berita 1 dengan berita 3 : ", edit_distance(doc_dict[1], doc_dict[3]))
print("Berita 1 dengan berita 4 : ", edit_distance(doc_dict[1], doc_dict[4]))
print("Berita 1 dengan berita 5 : ", edit_distance(doc_dict[1], doc_dict[5]))
print("Berita 2 dengan berita 3 : ", edit_distance(doc_dict[2], doc_dict[3]))
print("Berita 2 dengan berita 4 : ", edit_distance(doc_dict[2], doc_dict[4]))
print("Berita 2 dengan berita 5 : ", edit_distance(doc_dict[2], doc_dict[5]))
print("Berita 3 dengan berita 4 : ", edit_distance(doc_dict[3], doc_dict[4]))
print("Berita 3 dengan berita 5 : ", edit_distance(doc_dict[3], doc_dict[5]))
print("Berita 4 dengan berita 5 : ", edit_distance(doc_dict[4], doc_dict[5]))

def jaccard_sim(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

print("\nSkor Kemiripan dengan Jaccard Similarity")
print("Berita 1 dengan berita 2 : ", jaccard_sim(doc_dict[1].split(" "), doc_dict[2].split(" ")))
print("Berita 1 dengan berita 3 : ", jaccard_sim(doc_dict[1].split(" "), doc_dict[3].split(" ")))
print("Berita 1 dengan berita 4 : ", jaccard_sim(doc_dict[1].split(" "), doc_dict[4].split(" ")))
print("Berita 1 dengan berita 5 : ", jaccard_sim(doc_dict[1].split(" "), doc_dict[5].split(" ")))
print("Berita 2 dengan berita 3 : ", jaccard_sim(doc_dict[2].split(" "), doc_dict[3].split(" ")))
print("Berita 2 dengan berita 4 : ", jaccard_sim(doc_dict[2].split(" "), doc_dict[4].split(" ")))
print("Berita 2 dengan berita 5 : ", jaccard_sim(doc_dict[2].split(" "), doc_dict[5].split(" ")))
print("Berita 3 dengan berita 4 : ", jaccard_sim(doc_dict[3].split(" "), doc_dict[4].split(" ")))
print("Berita 3 dengan berita 5 : ", jaccard_sim(doc_dict[3].split(" "), doc_dict[5].split(" ")))
print("Berita 4 dengan berita 5 : ", jaccard_sim(doc_dict[3].split(" "), doc_dict[5].split(" ")))

def euclidian_dist(vec1, vec2):
    temp = vec1 - vec2
    sum_sq = np.dot(temp.T, temp)
    return np.sqrt(sum_sq)

print("\nSkor Kemiripan dengan Euclidian Distance")
print("Berita 1 dengan berita 2 : ", euclidian_dist(TD[:, 0], TD[:, 1]))
print("Berita 1 dengan berita 3 : ", euclidian_dist(TD[:, 0], TD[:, 2]))
print("Berita 1 dengan berita 4 : ", euclidian_dist(TD[:, 0], TD[:, 3]))
print("Berita 1 dengan berita 5 : ", euclidian_dist(TD[:, 0], TD[:, 4]))
print("Berita 2 dengan berita 3 : ", euclidian_dist(TD[:, 1], TD[:, 2]))
print("Berita 2 dengan berita 4 : ", euclidian_dist(TD[:, 1], TD[:, 3]))
print("Berita 2 dengan berita 5 : ", euclidian_dist(TD[:, 1], TD[:, 4]))
print("Berita 3 dengan berita 4 : ", euclidian_dist(TD[:, 2], TD[:, 3]))
print("Berita 3 dengan berita 5 : ", euclidian_dist(TD[:, 2], TD[:, 4]))
print("Berita 4 dengan berita 5 : ", euclidian_dist(TD[:, 3], TD[:, 4]))

import math
def cosine_sim(vec1, vec2):
    vec1 = list(vec1)
    vec2 = list(vec2)
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)

print("\nSkor Kemiripan dengan Cosine Similarity")
print("Berita 1 dengan berita 2 : ", cosine_sim(TD[:, 0], TD[:, 1]))
print("Berita 1 dengan berita 3 : ", cosine_sim(TD[:, 0], TD[:, 2]))
print("Berita 1 dengan berita 4 : ", cosine_sim(TD[:, 0], TD[:, 3]))
print("Berita 1 dengan berita 5 : ", cosine_sim(TD[:, 0], TD[:, 4]))
print("Berita 2 dengan berita 3 : ", cosine_sim(TD[:, 1], TD[:, 2]))
print("Berita 2 dengan berita 4 : ", cosine_sim(TD[:, 1], TD[:, 3]))
print("Berita 2 dengan berita 5 : ", cosine_sim(TD[:, 1], TD[:, 4]))
print("Berita 3 dengan berita 4 : ", cosine_sim(TD[:, 2], TD[:, 3]))
print("Berita 3 dengan berita 5 : ", cosine_sim(TD[:, 2], TD[:, 4]))
print("Berita 4 dengan berita 5 : ", cosine_sim(TD[:, 3], TD[:, 4]))