"""
* kelime tokenizasyonu
* cümle tokenizasyonu

pip install nltk
"""

import nltk   #natural language toolkit
nltk.download("punkt")   #kelime ve cümle tokenizasyonu için gerekli olan veri 


#örnek text
raw_text = "Natural Language Processing is a fascinating field. It combines linguistics and computer science."


#kelime tokenizasyonu
word_tokens = nltk.word_tokenize(raw_text)
print(f"Kelime tokenizasyonu: {word_tokens}")   


#cümle tokenizasyonu
sentence_tokens = nltk.sent_tokenize(raw_text)
print(f"Cümle tokenizasyonu: {sentence_tokens}")
