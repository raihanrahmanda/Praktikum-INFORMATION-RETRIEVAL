# PRE-PROCESSING
import os

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return(f.read())

# Membaca semua berita , dan menyimpannya dalam array
# iterate through all file
def get_berita(path) :
    berita=[]
    for file in os.listdir(path):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            # call read text file function
            berita.append(read_text_file(file_path))
    return berita

# Case folding dan replace "-" ke huruf kecil
def case_folding(berita):
    beritas=[]
    for isi in berita:
        isi = isi.lower()
        isi = isi.replace("-"," ")
        beritas.append(isi)
    return beritas

# Definisi stemming dan tokenisasi
def tokenisasi(text):
    tokens = text.split(" ")
    return tokens

def stemming(text):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # stemming process
    output = stemmer.stem(text)
    return output

# Melakukan tokenisasi dan stemming pada semua berita
def stemming_sentence(text):
    output = ""
    for token in tokenisasi(text):
        output = output + stemming(token) + " "
    return output[:-1]

def make_doc_dict(berita):
    doc_dict = {}
    i = 0
    for doc in berita:
        doc_dict["berita"+str(i+1)] = stemming_sentence(doc)
        i += 1
    return doc_dict


# Inverted index
def make_inverted_index(doc_dict):
    vocab = []
    inverted_index = {}
    for doc_id,doc in doc_dict.items():
        for token in tokenisasi(doc):
            if token not in vocab:
                vocab.append(token)
                inverted_index[token] = []
            if token in inverted_index:
                if doc_id not in inverted_index[token]:
                    inverted_index[token].append(doc_id)
    return(vocab,inverted_index)

# Nomor 1
# Term Weighting
def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_docs[doc_id][word] = doc.count(word)
    return tf_docs

# Document frequency and inverse document frequency
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
        idf[word] = 1 + np.log((length + 1) / (doc_fre[word]+1))
    return idf


# TF * IDF score
def tfidf(vocab,tf,idf_scr,doc_dict):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr


# Term - Document Matrix
def make_TD (vocab, doc_dict,tf_idf):
    TD = np.zeros((len(vocab), len(doc_dict)))
    # Membuat dictionary untuk mengindeks kata dalam vocab
    word_index = {word: index for index, word in enumerate(vocab)}
    for word in vocab:
        for doc_id, doc in tf_idf.items():
            ind1 = word_index[word]
            ind2 = list(tf_idf.keys()).index(doc_id)
            TD[ind1][ind2] = tf_idf[doc_id][word]
    return(TD)

# TF_QUERY
def termFrequency(vocab, query):
    tf_query = {}
    for word in vocab:
        tf_query[word] = query.count(word)
    return tf_query


# Term - Query Matrix
def make_TQ(vocab, tf_query,idf):
    import numpy as np
    TQ = np.zeros((len(vocab), 1)) #hanya 1 query
    for word in vocab:
        ind1 = vocab.index(word)
        TQ[ind1][0] = tf_query[word]*idf[word]
    return TQ

# Cosine Similarity
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

# Mengambil top 3 document
from collections import OrderedDict
def exact_top_k(doc_dict, TD, q, k):
    relevance_scores = {}
    i = 0
    for doc_id in doc_dict.keys():
        relevance_scores[doc_id] = cosine_sim(q, TD[:, i])
        i = i + 1
    sorted_value = OrderedDict(sorted(relevance_scores.items(),key=lambda x: x[1], reverse = True))
    top_k = {j: sorted_value[j] for j in list(sorted_value)[:k]}
    return top_k

# Nomor 1
def main1():
    print("\n")
    import time as tm
    path = "C:/Users/user/Downloads/berita"
    berita = get_berita(path)
    pre_process = case_folding(berita)
    doc_dict = make_doc_dict(pre_process)
    vocab,inverted_index = make_inverted_index(doc_dict)
    TFinDocs = termFrequencyInDoc(vocab, doc_dict)   
    idf = inverseDocFre(vocab,wordDocFre(vocab,doc_dict),len(doc_dict))
    tf_idf = tfidf(vocab,TFinDocs,idf,doc_dict)
    TD = make_TD(vocab, doc_dict,tf_idf)
    query = "vaksin corona jakarta"
    tf_query = termFrequency(vocab, query)
    TQ = make_TQ(vocab, tf_query, idf)
    start = tm.time()
    top_3 = exact_top_k(doc_dict, TD, TQ[:,0], 3)
    end = tm.time()
    print("No Index Elimination :")
    print("Metode : Cosine Similarity")
    print("Query : vaksin corona jakarta")
    print("TOP 3 document : ", top_3)
    print("Durasi : ",end - start)
main1()

# Nomor 2
def index_elim_simple(query, doc_dict):
    remove_list =[]
    for doc_id,doc in doc_dict.items():
        n = 0
        for word in tokenisasi(query):
            if stemming(word) in doc:
                n = n+1
        if n==0:
            remove_list.append(doc_id)
    for key in remove_list:
        del doc_dict[key]
    return doc_dict
    
def main2():
    import time as tm
    print("\n")
    path = "C:/Users/user/Downloads/berita"
    berita = get_berita(path)
    pre_process = case_folding(berita)
    query = "vaksin corona jakarta"
    doc_dict = index_elim_simple(query,make_doc_dict(pre_process))
    vocab,inverted_index = make_inverted_index(doc_dict)
    TFinDocs = termFrequencyInDoc(vocab, doc_dict)   
    idf = inverseDocFre(vocab,wordDocFre(vocab,doc_dict),len(doc_dict))
    tf_idf = tfidf(vocab,TFinDocs,idf,doc_dict)
    TD = make_TD(vocab, doc_dict,tf_idf)
    tf_query = termFrequency(vocab, query)
    TQ = make_TQ(vocab, tf_query, idf)
    start = tm.time()
    top_3 = exact_top_k(doc_dict, TD, TQ[:,0], 3)
    end = tm.time()
    print("Index Elimination :")
    print("Metode : Cosine Similarity")
    print("Query : vaksin corona jakarta")
    print("TOP 3 document : ", top_3)
    print("Durasi : ",end - start)

main2()