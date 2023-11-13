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
    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    stemmed_text = stemmer.stem(text)
    
    # Tokenize the stemmed text, and eliminate stopwords
    doc = nlp(stemmed_text)
    tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    
    return tokens

# Create an inverted index
inverted_index = {}

# List all files in the directory
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)) and file.endswith(".txt"):
        file_path = os.path.join(path, file)

        # Read the text from the file
        text = read_text_file(file_path)

        # Preprocess the text (stemming, tokenize, and eliminate stopwords)
        cleaned_tokens = preprocess_text(text)

        # Update the inverted index
        for token in set(cleaned_tokens):  # Use set to avoid duplicate documents
            inverted_index.setdefault(token, []).append(file)



def AND_optimized(postings):
    if not postings:
        return []
    
    postings.sort(key=len)
    result = postings[0]

    for i in range(1, len(postings)):
        current_postings = postings[i]
        new_result = []

        ptr1, ptr2 = 0, 0

        while ptr1 < len(result) and ptr2 < len(current_postings):
            if result[ptr1] == current_postings[ptr2]:
                new_result.append(result[ptr1])
                ptr1 += 1
                ptr2 += 1
            elif result[ptr1] < current_postings[ptr2]:
                ptr1 += 1
            else:
                ptr2 += 1

        result = new_result
        if not result:
            break

    return result

search_terms = []
while True:
    try:
        num_terms = int(input("Masukkan jumlah term yang akan diinput: "))
        break
    except ValueError:
        print("Masukkan angka yang valid untuk jumlah term.")


for i in range(num_terms):
    search_term = input("Masukkan term pencarian ke-{0}: ".format(i + 1))
    search_terms.append(search_term.strip())

posting_lists = [inverted_index.get(term, []) for term in search_terms]

search_result = AND_optimized(posting_lists)

if search_result:
    print("Hasil pencarian untuk terms:", search_terms)
    for file in search_result:
        print("Term ditemukan pada file:", file)
else:
    print("Tidak ada hasil pencarian yang cocok dengan terms:", search_terms)

