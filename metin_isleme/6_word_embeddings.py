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
    "Türkiye'de Ankara ve Gaziantep'in yemekleri çok güzel. "
]

tokenize_cumleler = [simple_preprocess(c) for c in cumleler]
print(tokenize_cumleler)

#word2vec 
Word2Vec_model = Word2Vec(
    sentences=tokenize_cumleler, 
    vector_size=50, 
    window=5, 
    min_count=1, #en az 1 kez geçen kelimeleri  al
    sg=0, #skip-gram = 1
    )


#fast text
FastText_model = FastText(
    sentences=tokenize_cumleler,
    vector_size=50,
    window=5,
    min_count=1,
    sg=0,
    )


def plot_word_embeddings(model, baslik):
    
    #modelin kelime vektörlerini al
    word_vectors = model.wv

    #ilk 1000 kelimeyi al
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[w] for w in words]

    # PCA boyut indirgeme
    pca = PCA(n_components=3)
    word_vectors_3d = pca.fit_transform(vectors)

    # 3D görselleştirme
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    #noktaları çiz
    ax.scatter(word_vectors_3d[:, 0], word_vectors_3d[:, 1], word_vectors_3d[:, 2]) # x, y, z

    #kelimeleri bu noktaların yanına yaz
    for i, word in enumerate(words):
        ax.text(word_vectors_3d[i, 0], word_vectors_3d[i, 1], word_vectors_3d[i, 2], word, fontsize = 13 )

    ax.set_title(baslik)
    ax.set_xlabel('Bileşen 1')
    ax.set_ylabel('Bileşen 2')
    ax.set_zlabel('Bileşen 3')
    plt.show()


#plot_word_embeddings(Word2Vec_model, "Word2Vec Gösterimi") 
plot_word_embeddings(FastText_model, "FastText Gösterimi")