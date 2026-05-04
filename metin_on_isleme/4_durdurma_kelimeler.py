"""
stop words çıakrma yöntemleri:

    -ingilizce stop words çıakrma
    -türkçe stop words çıakrma
    -manuel stop words çıkarma
"""

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")   #stop words veri tabanını indirme


stop_words_en = set(stopwords.words("english"))   #ingilizce stop words kümesi

#örnek text
eng_text = "This is just a simple example to show how stop words can be removed from sentences."
eng_text_list = eng_text.split()
print(eng_text_list)

filtered_words_eng = [word for word in eng_text_list if word.lower() not in stop_words_en]
print(f"İngilizce stop words kümesi: {stop_words_en}")
print(f"İngilizce stop words çıkarılmış metin: {filtered_words_eng}")



stop_words_tr = set(stopwords.words("turkish"))

#örnek text
tr_text = "Bu cümle, Türkçe stop words çıkarma işlemini göstermek için basit bir örnektir."
tr_text_list = tr_text.split()
#print(tr_text_list)

filtered_words_tr = [word for word in tr_text_list if word.lower() not in stop_words_tr]
print(f"Türkçe stop words kümesi: {stop_words_tr}")
print(f"Türkçe stop words çıkarılmış metin: {filtered_words_tr}")






#kütüphane kullanmadan manuel stop words çıkarma
custom_tr_stopwords = ["bu", "ile", "mi", "ki", "de", "da"]
custom_text = "Bu bir denemedir, bunun için amacımız metinlerde ki bazı kelimeleri çıkartmak."
custom_text_list = custom_text.split()
filtered_custom_text = [word for word in custom_text_list if word.lower() not in custom_tr_stopwords]
print(f"Orijinal metin: {custom_text}")
print(f"Manuel stop words kümesi: {custom_tr_stopwords}")