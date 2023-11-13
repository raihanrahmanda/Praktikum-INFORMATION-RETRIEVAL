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

for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)) and file.endswith(".txt"):
        file_path = os.path.join(path, file)

        text = read_text_file(file_path)

        cleaned_tokens = preprocess_text(text)

        # Update the inverted index
        for token in set(cleaned_tokens):  # Use set to avoid duplicate documents
            inverted_index.setdefault(token, []).append(file)

def AND(posting1, posting2=None, posting3=None):
    p1, p2, p3 = 0, 0, 0
    result = []

    if posting2 is None and posting3 is None:
        while p1 < len(posting1):
            result.append(posting1[p1])
            p1 += 1
        
    elif posting3 is None:
        while p1 < len(posting1) and p2 < len(posting2):
            if posting1[p1] == posting2[p2]:
                result.append(posting1[p1])
                p1 += 1
                p2 += 1
            elif posting1[p1] > posting2[p2]:
                p2 += 1
            else:
                p1 += 1
    else:
        while p1 < len(posting1) and p2 < len(posting2) and p3 < len(posting3):
            if posting1[p1] == posting2[p2] == posting3[p3]:
                result.append(posting1[p1])
                p1 += 1
                p2 += 1
                p3 += 1
            else:
                min_val = min(posting1[p1], posting2[p2], posting3[p3])
                if posting1[p1] == min_val:
                    p1 += 1
                if posting2[p2] == min_val:
                    p2 += 1
                if posting3[p3] == min_val:
                    p3 += 1

    return result

def OR(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            result.append(posting2[p2])
            p2 += 1
        else:
            result.append(posting1[p1])
            p1 += 1
    while p1 < len(posting1):
        result.append(posting1[p1])
        p1 += 1
    while p2 < len(posting2):
        result.append(posting2[p2])
        p2 += 1
    return result

def NOT(term, inverted_index):
    matching_documents = set()

    if term in inverted_index:
        matching_documents.update(inverted_index[term])

    all_documents = set()
    for documents in inverted_index.values():
        all_documents.update(documents)

    not_matching_documents = all_documents - matching_documents

    result = sorted(not_matching_documents)

    return result

hasil1 = AND(inverted_index['corona'])
print("term 'corona' berada pada file:", hasil1)

hasil2 = AND(inverted_index['covid'])
print("term 'covid' berada pada file:", hasil2)

hasil3 = AND(inverted_index['vaksin'])
print("term 'vaksin' berada pada file:", hasil3)

hasil4 = OR(inverted_index['corona'],inverted_index['covid'])
print("term 'corona' OR 'covid' berada pada file:", hasil4)

hasil5 = AND(inverted_index['vaksin'],inverted_index['corona'])
print("term 'vaksin' AND 'corona' berada pada file", hasil5)

hasil6 = AND(inverted_index['vaksin'],inverted_index['corona'],inverted_index['pfizer'])
print("term 'vaksin' AND 'corona' AND 'pfizer' berada pada file", hasil6)

hasil7 = NOT('vaksin', inverted_index)
print(f"term 'vaksin' tidak berada pada file: {hasil7}")


# def NOT(query, inverted_index):
#     # Initialize a set to store the matching documents
#     matching_documents = set()

#     # Check if the query is a single term
#     if query in inverted_index:
#         # Add documents containing the query term to the matching documents set
#         matching_documents.update(inverted_index[query])

#     # Get the set of all documents in the inverted index
#     all_documents = set()
#     for documents in inverted_index.values():
#         all_documents.update(documents)

#     # Calculate the set of documents that do not match the query
#     not_matching_documents = all_documents - matching_documents

#     # Sort the not_matching_documents list in ascending order (by file name)
#     result = sorted(not_matching_documents)

#     return result