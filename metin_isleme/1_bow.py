"""
Amaç: 
    - metin temsili(bag of words): metin listesi -> sayısal vektör
    - sklearn CountVectorizer sınıfı: kelimelerin kaç defa geçtiğini sayar ve vektör temsili oluşturur

Sonuç:
    - kelime kümesi (vocabulary)
    - her metin listesi sayısal vektörler ile temsil edilir
"""

from sklearn.feature_extraction.text import CountVectorizer

dokumanlar = [
    "kedi bahçede",
    "kedi evde"
]

# CountVectorizer nesnesi oluşturulur
kelime_sayac = CountVectorizer()

# dokumanlar sayısal vektörlere dönüştürülür (bag of words uygulama)
dokuman_vektorleri = kelime_sayac.fit_transform(dokumanlar)

#countvectorizer nesnesi ile bulunan listesi (vocabulary)
kelime_kumesi = kelime_sayac.get_feature_names_out()
print(f"kelime kümesi: {kelime_kumesi}")
print(dokuman_vektorleri.toarray()) 