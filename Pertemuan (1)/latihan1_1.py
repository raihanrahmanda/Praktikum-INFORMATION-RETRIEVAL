# Import Module
import os

# Folder Path
path = "D:/RAIHAN STIS/Perkuliahan/SEMESTER 5/Praktikum INFORMATION RETRIEVAL/Pertemuan (1)/berita"

# List all files in a directory using os.listdir
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        print(file)


print("\n=====================================")
print("isi dari berita.txt")
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

# iterate through all file
for file in os.listdir(path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
        # call read text file function
        read_text_file(file_path)


print("\n=====================================")
text_1 = "Wilayah Kamu Sudah 'Bebas' COVID-19? Cek 34 Kab/Kota Zona Hijau Terbaru"
text_2 = "Vaksin COVID-19 Bakal Rutin Setiap Tahun? Tergantung, Ini Penjelasannya"
text_3 = "RI Mulai Suntikkan Booster di 2022, Masihkah Ampuh Lawan Varian Delta Cs?"
query = "COVID-19"
docs = [text_1, text_2, text_3]
for doc in docs:
    if query in doc:
        print(doc)


#displays a list of document names in the "news" folder which contains the query "corona".
print("\n=====================================")
path = "D:/RAIHAN STIS/Perkuliahan/SEMESTER 5/Praktikum INFORMATION RETRIEVAL/Pertemuan (1)/berita"
query = "corona"
for file in os.listdir(path):
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
        with open(file_path, 'r') as f:
            text = f.read()
            if query in text:
                print(file)

                
