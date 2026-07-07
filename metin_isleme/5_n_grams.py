"""
Verilen örnek cümleler üzerinden N Gram (unigram, bigram, trigram) analizi gerçekleştirmek.

* örnek belgeleri tanımla
* CountVectorizer ile unigram, bigram ve trigram örnekleri oluştur.
* her bir model için özellik (kelime/ kelime grubu) çıkartılır.
* sonuçları ekrana yazdır.
"""

from sklearn.feature_extraction.text import CountVectorizer

belgeler = [
    "Bu bir örnek cümledir.",
    "N gram analizi metin madenciliğinde kullanılır."
    ]

unigram_model = CountVectorizer(ngram_range=(1, 1))
bigram_model = CountVectorizer(ngram_range=(2, 2))
trigram_model = CountVectorizer(ngram_range=(3, 3))                         


x_unigram = unigram_model.fit_transform(belgeler) #sayısal vektörlere dönüştürme
unigram_ozellikler = unigram_model.get_feature_names_out() #kelime listesi

x_bigram = bigram_model.fit_transform(belgeler) #sayısal vektörlere dönüştürme
bigram_ozellikler = bigram_model.get_feature_names_out() #kelime listesi

x_trigram = trigram_model.fit_transform(belgeler) #sayısal vektörlere dönüştürme
trigram_ozellikler = trigram_model.get_feature_names_out() #kelime listesi


print(f"Unigram özellikler: {unigram_ozellikler}")
print(f"Bigram özellikler: {bigram_ozellikler}")
print(f"Trigram özellikler: {trigram_ozellikler}")