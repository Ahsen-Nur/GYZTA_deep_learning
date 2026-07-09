"""
Küçük bir veri seti üzerinden Word embedding sonra PCA ile görselleşitrme 
    *word2: google
    *fasttext: meta

* örnek cümle veri seti oluştur. 
* preprocessing
* word2vec ve fasttext modelleri eğitilir
* her iki modelden elde edilen vektörleri PCA analizi ile 3 boyuta indirgenir
* kelime vektörlerini 3 boyutlu olarak görselleştir
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess


cumleler = [
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi severler.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır.",
    "Türkiye'nin başkenti Ankara'dır.",
    "Gaziantep yemekleri çok güzel. "
]

tokenize_cumleler = [simple_preprocess(c) for c in cumleler]
print(tokenize_cumleler)
