"""
Amaç:
*rnn kullanılarak duygu analizi(sentiment analysis) yapmak
*sınıflandırma problemi: restoran yorumları, etiket(olumlu/olumsuz)

Adımlar:
*kütüphanelerin yüklenmesi(tensorflow mu keras mı pytorch mu)
*restoran yorumlarını içeren veri setini yükle (simülasyon verisi oluşturma(gpt, gemini, drop))
*metin ön işleme(tokenization, padding, label encoding, train-test split) 
*embedding: word2vec ile sayısal vektörlere dönüştür
*rnn modeli oluşturma(embedding -> simpleRNN -> Dense layer)
*modelin derlenmesi ve eğitimi 
*test setinde modelin performansını değerlendirme
*user test(yeni cümlelerin sınıflandırması için fonksiyon tanımlama)
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer


from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#veri seti oluştur (restoran yorum ve etiketi)
rewiews_data = {
    "text": [
        "Yemekler harikaydı, her şey taze ve lezzetliydi.",
        "Garson çok ilgisizdi, siparişimi unuttular.",
        "Mekanın atmosferi çok sıcak, kesinlikle tekrar geleceğim.",
        "Fiyatlar porsiyonlara göre inanılmaz pahalı. Değmez.",
        "Etler tam kıvamında pişmişti, şefin ellerine sağlık.",
        "Masalar çok pisti, temizlenmesi için 15 dakika bekledik.",
        "Deniz ürünleri çok taze, mezeler efsaneydi.",
        "Çorba buz gibi geldi, garsona söylediğimizde ters bir tepki aldık.",
        "Ailecek gittik ve çok memnun kaldık, çocuk menüsü çok başarılı.",
        "Hesapta yediğimizden fazlası yazıyordu, itiraz edince zorla düzelttiler.",
        "Kahvaltı tabağı çok doyurucu ve çeşitliydi. Çay sınırsızdı.",
        "Müzik sesi o kadar yüksekti ki masada birbirimizi duyamadık.",
        "Vegan seçeneklerin bu kadar bol ve lezzetli olması harika.",
        "Salatanın içinden böcek çıktı, midemiz bulandı.",
        "Manzara şahane, şarap menüsü oldukça geniş ve kaliteli.",
        "Vale hizmeti çok kötüydü, arabamı teslim almak için yarım saat bekledim.",
        "Hızlı servis ve güler yüzlü çalışanlar. Öğle arası için ideal.",
        "Köfteler içi çiğ kalmış şekilde servis edildi, yiyemedik.",
        "Tatlılar muazzam! Özellikle sufle için bile gidilir.",
        "Rezervasyonumuz olmasına rağmen bizi kapıda 20 dakika beklettiler.",
        "İkramlar için çok teşekkür ederiz, çok ince bir davranıştı.",
        "Pizzanın hamuru yanmıştı ve malzemesi çok azdı.",
        "Yöresel yemekleri aslına çok uygun yapmışlar, tebrikler.",
        "Tuvaletler çok bakımsız ve sabun bile yoktu.",
        "Hem göze hem damağa hitap eden harika sunumlar vardı.",
        "İçecekler asitsiz ve ılıktı, hiç keyif almadık.",
        "Fiyat/performans açısından şehirdeki en iyi mekanlardan biri.",
        "Porsiyonlar o kadar küçüldü ki doyabilmek imkansız.",
        "Şef masamıza gelip bizimle ilgilendi, çok misafirperverlerdi.",
        "Menüdeki çoğu şey kalmamıştı, mecburiyetten başka bir şey söyledik.",
        "Geleneksel tatları modern bir dokunuşla sunmaları çok başarılı.",
        "Masalar birbirine o kadar yakın ki yan masanın tüm sohbetini dinledik.",
        "Bahçesi çok ferah, yaz akşamları için harika bir alternatif.",
        "Kredi kartı geçmediğini hesabı öderken söylediler, çok mağdur olduk.",
        "Makarnalar el yapımı ve sosları efsane. Bayıldım.",
        "Garsonlar kendi aralarında şakalaşmaktan müşterilerle ilgilenmiyorlar.",
        "Samimi, küçük ve lezzeti büyük bir esnaf lokantası.",
        "Et çok sertti, çiğnemekten çenemiz yoruldu.",
        "Sushiler çok başarılı, malzemeler birinci sınıf.",
        "İnternetteki menü fiyatlarıyla mekandaki fiyatlar arasında uçurum var.",
        "Kahvesi çok kaliteli, tatlıların yanında mükemmel gidiyor.",
        "Tavuktan tuhaf bir koku geliyordu, risk almamak için yemedim.",
        "Otopark sorunu olmaması büyük avantaj, yemekler de keza çok iyi.",
        "Çay o kadar bayattı ki renginden bile belli oluyordu.",
        "Hamburgerin ekmeği yumuşacık, eti çok suluydu. Kesinlikle tavsiye ederim.",
        "Klima doğrudan masamıza vuruyordu, kapatmalarını istedik ama yapmadılar.",
        "Çalışanlar menü konusunda çok bilgili ve yönlendirmeleri isabetliydi.",
        "Bardaklar lekeliydi, temizini istemek zorunda kaldık.",
        "Doğum günü organizasyonumuz için kusursuz bir hizmet verdiler.",
        "Çok sıra vardı, ancak beklediğimize kesinlikle değmedi, lezzet vasattı."
    ],
    "label": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}


#dataframe oluştur
df = pd.DataFrame(rewiews_data)
print(df.head())

#metin ön işleme
#tokenization
tokenizer = Tokenizer() #Metindeki her benzersiz kelimeye bir tam sayı (indeks) atar
tokenizer.fit_on_texts(df['text']) #Tüm yorumları tarar. Kelime frekans sözlüğü oluşturur: Hangi kelime kaç kez geçmiş
text_sequences = tokenizer.texts_to_sequences(df['text']) #Artık her yorum, kelime indekslerinin bir listesine dönüşür. Kelime sırası korunur; bu TF-IDF'ten en temel farktır.
word_index = tokenizer.word_index # sözlük: kelime -> index
print(word_index)


#padding
max_sequence_length = max(len(seq) for seq in text_sequences) #en uzun yorumun uzunluğunu bulur
print(f"En uzun yorumun uzunluğu: {max_sequence_length}") 

X = pad_sequences(text_sequences, maxlen=max_sequence_length) #tüm yorumları aynı uzunlukta olacak şekilde sıfırlarla doldurur
print(f"giriş verisinin boyutu: {X.shape}") #giriş verisinin boyutu: (yorum sayısı, en uzun yorumun uzunluğu)
print(X)


#label encoding
label_encoder = LabelEncoder() #etiketleri sayısal değerlere dönüştürür
y = label_encoder.fit_transform(df['label']) #etiketleri sayısal değerlere dönüştürür


#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


#embedding
sentences = [text.split() for text in df['text']] #Her yorumu boşluklardan bölüp kelime listesine çevirir. Word2Vec ham metin değil, token listesi ister

#word2vec modelini eğit
word2vec_model = Word2Vec(sentences, 
                          vector_size=50, 
                          window=5, #Bir kelimenin bağlam penceresi: sağda ve solda 5'er komşu kelimeye bakarak ilişki öğrenir
                          min_count=1 #Sadece 1 kez geçen kelimeleri bile dahil et
                          )

embedding_dim = 50 #her kelime 50 boyutlu vektörle temsil edilecek

#embedding matrisini oluştur
embedding_matrix = np.zeros((len(word_index) + 1, #Tokenizer 1'den indekslemeye başlar. 0 indeksi padding için ayrıktır. O yüzden matrisin satır sayısı toplam kelime sayısı + 1'dir. İlk satır (0) sıfırlarla doldurulur.
                             embedding_dim
                             )) #sıfırlarla doldurulmuş bir matris oluşturur

for word, idx in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[idx] = word2vec_model.wv[word] 

print(f"embedding matrix: {embedding_matrix}")


#model oluşturma
model = Sequential()

#embedding katmanı
model.add(Embedding(input_dim=len(word_index) + 1, #sözlükteki toplam kelime sayısı +1
                    output_dim=embedding_dim, #embedding boyutu
                    weights=[embedding_matrix], #önceden eğitilmiş word2vec embedding matrisi
                    input_length=max_sequence_length, #cümlelerin uzunluğu
                    trainable=False #embedding ağırlıkları sabit kalacak
                    )) 


#rnn katmanı: units-> gizli katman sayısı, return_sequences-> sadece son çıktıyı return eder. son hafıza durumunu verir.
model.add(SimpleRNN(units = 50, return_sequences = False))

#output katmanı
model.add(Dense(1, activation= "sigmoid"))

#compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#training
model.fit(X_train, y_train,
          epochs = 10, #Tüm eğitim verisini 10 kez baştan sona oku. Her turda ağırlıklar güncellenir
          batch_size = 2, #Her seferinde 2 örnek al, hata hesapla, ağırlıkları güncelle.
          validation_data = (X_test, y_test) #her epoch sonunda test seti ile doğrulama
          ) 


#değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


#yeni cümlelerin sınıflandırması için fonk.
def classify_sentence(sentence):
    """
    yeni cümleyi alır, işleme sokar ve model ile sınıflandırmaya çalışır
    """

    seq = tokenizer.texts_to_sequences([sentence]) #cümleyi sayısal dizilere çevirir
    padded_seq = pad_sequences(seq, maxlen = max_sequence_length) #cümleyi sayısal diziler çevirir

    prediction = model.predict(padded_seq) #modelden olasılık al
    prediced_class = (prediction > 0.5).astype(int) #0.5 üstü positive etiket alır

    label = "positive" if prediced_class[0][0] == 1 else "negative"
    return label


new_sentence = "restoran çok temizdi ve yemekler çok güzeldi"
result = classify_sentence(new_sentence)
print(result)
