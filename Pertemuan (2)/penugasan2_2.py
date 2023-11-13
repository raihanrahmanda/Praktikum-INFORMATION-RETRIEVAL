# Import Module
import os
from spacy.lang.id import Indonesian
import spacy
from spacy.lang.id.stop_words import STOP_WORDS
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nlp = Indonesian()  # Load Indonesian language model

# Folder Path
path = "D:/RAIHAN STIS/Perkuliahan/SEMESTER 5/Praktikum INFORMATION RETRIEVAL/Pertemuan (2)/berita"

# Function to read text file
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # content = content.replace('\n', ' ')  # Replace newline with space
    return content

print("\n============================================================\n")
print("Isi dari folder berita")
print("\n============================================================\n")
# List all files in a directory using os.listdir
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        print(file)


print("\n============================================================\n")
print("isi dari berita.txt")
print("\n============================================================\n")
# iterate through all file
for file in os.listdir(path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        #print kalimat berita ke-i
        print("isi dari", file, ":")
        file_path = f"{path}\{file}"
        # call read text file function
        result1 = read_text_file(file_path)
        print(result1)
        print("\n")

print("\n============================================================\n")
print("Case Folding masing-masing isi dari folder berita")
print("\n============================================================\n")
# iterate through all file
for file in os.listdir(path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        #print kalimat berita ke-i
        print("isi dari", file, ":")
        file_path = f"{path}\{file}"
        # call read text file function
        result2 = read_text_file(file_path)
        print("teks original:")
        print(result2)
        print("\n")
        print("teks case folding (lowercase):")
        print(result2.lower())
        print("\n")
        print("teks case folding (uppercase):")
        print(result2.upper())
        print("\n")

print("\n============================================================\n")
print("Tokenisasi masing-masing isi dari folder berita")
print("\n============================================================\n")
def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        doc = nlp(content)
        tokens = [token.text for token in doc]
        return tokens

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        tokens = tokenize_file(file_path)
        print(f"Tokens in {filename}: {len(tokens)}")
        print(tokens) 

print("\n============================================================\n")
print("Stopword elimination masing-masing isi dari folder berita")
print("\n============================================================\n")
def eliminate_stopwords(tokens):
    cleaned_tokens = [token for token in tokens if token.lower() not in STOP_WORDS]
    return cleaned_tokens

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        tokens = tokenize_file(file_path)
        tokens_without_stopwords = eliminate_stopwords(tokens)
        
        print(f"Tokens in {filename} after stopwords elimination: {len(tokens_without_stopwords)}")
        print(tokens_without_stopwords)
        print("-" * 60)  # Add a separator between outputs

print("\n============================================================\n")
print("Stemming masing-masing isi dari folder berita")
print("\n============================================================\n")
stemmer = StemmerFactory().create_stemmer()

for file in os.listdir(path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        #print kalimat berita ke-i
        print("isi dari", file, "setelah stemming:")
        file_path = f"{path}\{file}"
        # call read text file function
        result3 = read_text_file(file_path)
        print(stemmer.stem(result3))
        print("\n")