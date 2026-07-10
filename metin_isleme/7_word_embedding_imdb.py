"""
- imdb film yorumları üzerinden word2vec tabanlı kelime vektörleri üretmek
ve kmeans algoritmasını kullanarak kümelere ayırma

*veriseti yükleme
*metin temizleme 
*tokenization
*word2vec modelini tanımla ve embedding gerçekleştir
*ilk 500 kelime kmeans ile 2li kümeleme ayır (pozitif ve negatif yorumlar)
*pca ile 50 boyuttan 2 boyuta indir
*sonuçları 2 boyutta görselleştir
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


veri = pd.read_csv("IMDB Dataset.csv")

yorumlar = veri["review"]
print(veri.head())


def temizle(metin):
    metin = metin.lower()
    metin = re.sub(r"\d+", " ", metin)
    metin = re.sub(r"[^\w\s]", " ", metin)
    metin = " ".join([kelime for kelime in metin.split() if len(kelime) > 2]) #çok kısa kelimeleri çıkar
    #stopwords çıkarma
    metin = " ".join(
        [word for word in metin.split()
         if word not in ENGLISH_STOP_WORDS]
    )

    return metin

temizlenmis_yorumlar = [temizle(y) for y in yorumlar]
print(temizlenmis_yorumlar[:5])


#tokenization
tokenized_yorumlar = [simple_preprocess(y) for y in temizlenmis_yorumlar]

#word2vec modeli
Word2Vec_model = Word2Vec(
    sentences=tokenized_yorumlar, 
    vector_size=2, #50 
    window=5, 
    min_count=1,
    sg=0
)

kelime_vektorleri = Word2Vec_model.wv

#ilk 500 kelime ve vektörlerini al
kelimeler = list(kelime_vektorleri.index_to_key)[:500]  #ilk 500 kelime
vektorler = [kelime_vektorleri[w] for w in kelimeler]
vektorler_np = np.array(vektorler)

#kmeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(vektorler) #training
kume_etiketleri = kmeans.labels_ #her kelime için küme etiketleri

#pca ile boyut indirgeme
#pca = PCA(n_components=2)
#vektorler_2d = pca.fit_transform(vektorler)

#görselleştirme
plt.figure()
#plt.scatter(vektorler_2d[:, 0], vektorler_2d[:, 1], c=kume_etiketleri, cmap='viridis')
plt.scatter(vektorler_np[:,0], vektorler_np[:,1], c=kume_etiketleri, cmap='viridis')


#kume merkezlerini işaretle
#merkezler_2d = pca.transform(kmeans.cluster_centers_)
#plt.scatter(merkezler_2d[:, 0], merkezler_2d[:, 1], c='red', marker='X', s=150, label='merkezler')
merkezler = kmeans.cluster_centers_
plt.scatter(merkezler[:,0], merkezler[:,1], c='red', marker='X', s=150, label='merkezler')


#kelimeleri noktalar üzerine yaz
for i, kelime in enumerate(kelimeler):
    #plt.text(vektorler_2d[i, 0], vektorler_2d[i, 1], kelime, fontsize=9)
    plt.text(vektorler_np[i,0], vektorler_np[i,1], kelime, fontsize=9)


plt.title("Word2Vec + KMeans Kümeleme")
plt.legend()
plt.show()


"""
vector_size=2 ve PCA uygulamasında n components=2 olarak ayarlandması arasındaki fark şudur:
Word2Vec her kelimeyi 2 boyutlu bir vektör ile temsil eder. Görselleştirme kolaydır çünkü 2D uzaydadır.
Ama bu kadar düşük boyut, kelimeler arasındaki karmaşık anlamsal ilişkileri yakalayamaz.
Word2Vec kelimeleri 50 boyutlu uzayda temsil eder. Bu uzayda semantik ilişkiler daha iyi yakalanır.
PCA ile bu 50 boyutu 2 boyuta indiriyoruz. PCA, en fazla varyansı koruyan eksenleri seçtiği için görselleştirme 
daha anlamlı olur. 

“Boyut” dediğimiz şey aslında embedding uzayındaki koordinat sayısıdır. Yani her kelimeyi temsil eden 
vektörün kaç bileşeni olduğudur. 
Kelime vektörleri: Word2Vec her kelimeyi bir sayı dizisiyle (vektör) temsil eder. Örneğin vector_size=50 
olduğundaher kelime 50 sayıdan oluşan bir vektör olur. Her boyut bir özellik taşır. 
Boyut arttıkça kapasite artar. Daha fazla boyut, kelimeler arasındaki karmaşık ilişkileri kodlamak için daha 
fazla yer sağlar.
2 boyut → sadece iki eksen, çok sınırlı ilişki.
50 boyut → daha fazla eksen, daha çok semantik ayrım.
300 boyut → çok zengin ilişkiler, örneğin “king – man + woman ≈ queen” gibi analogiler daha doğru çıkar.

PCA şu işi yapar:
Yüksek boyutlu vektörleri alır.
En fazla varyansı (bilgi içeriğini) koruyan eksenleri seçer.
Bu eksenleri 2D veya 3D’ye indirger.
Böylece görselleştirme yapılırken anlamlı ayrımlar korunur.

Varyans, bir veri kümesindeki değerlerin ne kadar farklılaştığını ölçer.
Yüksek varyans → değerler birbirinden uzak, çeşitlilik fazla. Bu çeşitlilik, verinin bilgi içeriğini taşır.
Düşük varyans → değerler birbirine yakın, çeşitlilik az.
"""