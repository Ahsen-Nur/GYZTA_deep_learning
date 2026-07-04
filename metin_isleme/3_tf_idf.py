"""
- küçük bir belge oluştur
- tf-idf vektorizer ile belgeleri sayısal vektörlere dönüştür
- kelime kümesini çıkart 
- belgelerin tf-idf vektör temsillerini elde et
- tüm belgeler için kelimelerin ortalama tf-idf değerlerini hesaplas
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

belgeler = [
    "köpek çok tatlı bir hayvandır",
    "köpek ve kuşlar çok tatlı hayvanlardır",
    "inekler süt üretirler",
    "köpek, köpek, köpek, köpek, köpek, köpek, köpek, köpek, köpek, köpek"
]

# TF-IDF işlemleri

# TF-IDF vektörizer nesnesi oluştur
tfidf_vectorizer = TfidfVectorizer()

# belgeleri sayısal vektörlere dönüştür
belge_vektörleri = tfidf_vectorizer.fit_transform(belgeler)

kelime_kümesi = tfidf_vectorizer.get_feature_names_out() #vocabulary

# belgelerin tf-idf vektör temsillerini elde et
vektör_temsili = belge_vektörleri.toarray()
print(f"TF-IDF matrisi:\n{vektör_temsili}\n")
df_tfidf = pd.DataFrame(vektör_temsili, columns=kelime_kümesi)
print(df_tfidf)

#her kelimenin belgeler arası ortalama tf-idf değeri
ortalama_tfidf = df_tfidf.mean(axis=0)
print(f"tf-idf ortalamaları:\n{ortalama_tfidf}")
