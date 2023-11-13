def paragraph_parsing(document):
    list_paragraf = document.split('\n')
    return list_paragraf

def sentence_parsing(paragraph):
    list_kalimat = [kalimat.strip() for kalimat in re.split(r'[.!?]', paragraph) if kalimat.strip()]
    return list_kalimat

# Contoh input dokumen
input_dokumen = """Mobilitas warga bakal diperketat melalui penerapan PPKM level 3 se-Indonesia di masa libur Natal dan tahun baru (Nataru). Rencana kebijakan itu dikritik oleh Epidemiolog dari Griffith University Dicky Budiman.
Dicky menyebut pembatasan mobilitas memang akan memiliki dampak dalam mencegah penularan COVID-19. Tapi, kata dia, dampaknya signifikan atau tidak akan bergantung pada konsistensi yang mendasar yakni testing, tracing, treatment (3T) hingga vaksinasi COVID-19."""

import re

# Memisahkan paragraf
list_paragraf = paragraph_parsing(input_dokumen)

# Menampilkan list paragraf
print("List paragraf:")
for idx, paragraf in enumerate(list_paragraf, start=1):
    print(f"p{idx}: {paragraf}")

# Memisahkan kalimat pada paragraf 1
paragraf_1 = list_paragraf[0]
list_kalimat_paragraf_1 = sentence_parsing(paragraf_1)

# Menampilkan list kalimat pada paragraf 1
print("\nList kalimat pada paragraf 1:")
for idx, kalimat in enumerate(list_kalimat_paragraf_1, start=1):
    print(f"s{idx}: {kalimat}")