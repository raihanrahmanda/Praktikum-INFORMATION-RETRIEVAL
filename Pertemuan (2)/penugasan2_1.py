doctask = '''Mobilitas warga bakal diperketat melalui penerapan
PPKM level 3 se-Indonesia di masa libur Natal dan
tahun baru (Nataru). Rencana kebijakan itu dikritik
oleh Epidemiolog dari Griffith University Dicky
Budiman.

Dicky menyebut pembatasan mobilitas memang akan
memiliki dampak dalam mencegah penularan COVID-19. 
Tapi, kata dia, dampaknya signifikan atau tidak akan
bergantung pada konsistensi yang mendasar yakni
testing, tracing, treatment (3T) hingga vaksinasi 
COVID-19.'''

# fungsi untuk memisahkan paragraf dalam suatu dokumen
def paragraph_parsing(text):
    paragraphs = text.split("\n\n")
    return paragraphs

# fungsi untuk memisahkan kalimat dalam paragraf hasil parsing
def sentence_parsing(text):
    sentences = text.split(". ")
    return sentences


list_paragraph = paragraph_parsing(doctask)
print("List paragraf:")
for i in range(len(list_paragraph)):
    print("P", i+1, ":", list_paragraph[i])
    print("\n")

print("===========================================================\n")

for i in range(len(list_paragraph)):
    list_sentence = sentence_parsing(list_paragraph[i])
    print("List kalimat pada paragraf ke-", i+1, ":\n")
    for j in range(len(list_sentence)):
        print("S", j+1, ":", list_sentence[j])
        print("\n")