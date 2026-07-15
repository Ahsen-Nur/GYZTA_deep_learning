"""
amaç: lstm tabanlı bir dil modeli ile metin üretimi(text generation) gerçekleştirme
eğitim verisi olarak gemini ile oluşturulmuş tr günlük ifadeler/cümleler oluştur.
model verilen bir başlangıç kelimesinden yeni kelimeler ya da cümleler üretir.

adımlar:
*eğitim verisini hazırlama
*tokenization: kelimeleri sayısal vektörlere çevir
*n-gram dizileri oluştur: dil modeli için girdi-çıktı çiftleri hazırlama
*padding: tüm dizileri aynı uzunluğa getirme
*lstm tabanlı model kurulumu
*model eğitimi
*yeni metin üretimi için fonk. yazılması
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#eğitim veriseti
texts=[
    "bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "kitap okumak beni gerçekten mutlu ediyor.",
    "kahvaltıda sıcak çay içmek güne iyi başlatıyor.",
    "arkadaşlarımla buluşmak için sabırsızlanıyorum.",
    "yeni tarifler denemek mutfağı daha eğlenceli hale getiriyor.",
    "spor salonuna gitmek bana enerji veriyor.",
    "film izlemek akşamları en sevdiğim aktivite.",
    "telefonumun şarjı yine çok hızlı bitiyor.",
    "alışverişe çıkmak bazen terapi gibi geliyor.",
    "müzik dinlemek ruh halimi hemen değiştiriyor.",
    "bugün biraz yorgun hissediyorum.",
    "işe yetişmek için acele etmem gerekiyor.",
    "kahve içmeden güne başlayamıyorum.",
    "yeni şeyler öğrenmek bana heyecan veriyor.",
    "tatilde deniz kenarında vakit geçirmek harika olurdu.",
    "yağmur yağınca evde kalmayı seviyorum.",
    "arkadaşım bana sürpriz yaptı, çok mutlu oldum.",
    "bugün çok işim var, zamanımı iyi kullanmalıyım.",
    "yeni bir dil öğrenmek istiyorum.",
    "bisiklet sürmek bana özgürlük hissi veriyor.",
    "akşam yemeğinde makarna yapmayı düşünüyorum.",
    "kitapçıya gitmek beni her zaman heyecanlandırıyor.",
    "bugün biraz tembellik yapmak istiyorum.",
    "çalışırken müzik dinlemek odaklanmamı kolaylaştırıyor.",
    "tatlı yemek bana keyif veriyor.",
    "yeni bir hobi edinmek istiyorum.",
    "arkadaşımın doğum günü için hediye aldım.",
    "bugün erken yatmayı planlıyorum.",
    "yeni bir film vizyona girmiş, izlemek istiyorum.",
    "alışveriş listemi hazırlamam gerekiyor.",
    "bahçede çiçeklerle ilgilenmek beni rahatlatıyor.",
    "bugün biraz kitap yazmayı deneyeceğim.",
    "yeni bir kahve dükkanı keşfettim.",
    "telefonumda çok fazla fotoğraf var, düzenlemeliyim.",
    "spor yapmak sağlığım için çok önemli.",
    "bugün biraz meditasyon yapmayı düşünüyorum.",
    "arkadaşım bana güzel bir mesaj attı.",
    "yeni bir müzik albümü çıktı, dinlemek istiyorum.",
    "bugün biraz temizlik yapmam gerekiyor.",
    "yeni bir tarif denemek için malzeme aldım.",
    "akşam yürüyüşü yapmak bana huzur veriyor.",
    "bugün biraz resim çizmeyi deneyeceğim.",
    "arkadaşlarımla oyun oynamak çok eğlenceli.",
    "yeni bir kitap sipariş ettim.",
    "bugün biraz dinlenmek istiyorum.",
    "yeni bir kahve makinesi aldım.",
    "alışveriş merkezine gitmek istiyorum.",
    "bugün biraz yazı yazmayı deneyeceğim.",
    "arkadaşım bana kahve ısmarladı.",
    "yeni bir spor ayakkabı aldım.",
    "bugün biraz müzik dinlemek istiyorum.",
    "yeni bir film izlemek için plan yaptım.",
    "alışveriş yapmak bana keyif veriyor.",
    "bugün biraz yürüyüş yapmayı düşünüyorum.",
    "arkadaşım bana kitap önerdi.",
    "yeni bir telefon almak istiyorum.",
    "bugün biraz kahve içmek istiyorum.",
    "yeni bir oyun denemek istiyorum.",
    "alışveriş listemi hazırladım.",
    "bugün biraz spor yapmayı düşünüyorum.",
    "arkadaşım bana yemek yaptı.",
    "yeni bir müzik dinlemek istiyorum.",
    "bugün biraz kitap okumak istiyorum.",
    "yeni bir film izlemek istiyorum.",
    "alışveriş yapmak bana mutluluk veriyor.",
    "bugün biraz kahve içmek istiyorum.",
    "arkadaşım bana sürpriz yaptı.",
    "yeni bir oyun oynamak istiyorum.",
    "bugün biraz yürüyüş yapmak istiyorum.",
    "yeni bir kitap okumak istiyorum.",
    "alışveriş yapmak bana keyif veriyor.",
    "bugün biraz müzik dinlemek istiyorum.",
    "arkadaşım bana hediye verdi.",
    "yeni bir film izlemek istiyorum.",
    "bugün biraz kahve içmek istiyorum.",
    "yeni bir oyun denemek istiyorum.",
    "alışveriş yapmak bana mutluluk veriyor.",
    "bugün biraz kitap okumak istiyorum.",
    "arkadaşım bana yemek yaptı.",
    "yeni bir müzik dinlemek istiyorum.",
    "bugün biraz yürüyüş yapmak istiyorum.",
    "yeni bir kitap sipariş ettim.",
    "alışveriş yapmak bana keyif veriyor.",
    "bugün biraz kahve içmek istiyorum.",
    "arkadaşım bana sürpriz yaptı.",
    "yeni bir oyun oynamak istiyorum.",
    "bugün biraz müzik dinlemek istiyorum.",
    "yeni bir film izlemek istiyorum.",
    "alışveriş yapmak bana mutluluk veriyor.",
    "bugün biraz kitap okumak istiyorum.",
    "arkadaşım bana hediye verdi.",
    "yeni bir oyun denemek istiyorum.",
    "bugün biraz yürüyüş yapmak istiyorum.",
    "yeni bir müzik dinlemek istiyorum.",
    "alışveriş yapmak bana keyif veriyor.",
    "bugün biraz kahve içmek istiyorum.",
    "arkadaşım bana yemek yaptı.",
    "yeni bir kitap okumak istiyorum.",
    "bugün biraz müzik dinlemek istiyorum.",
    "yeni bir film izlemek istiyorum.",
    "alışveriş yapmak bana mutluluk veriyor.",
]


#preprocessing
#tokenizer: her kelimeyi benzersiz bir indexe atar
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) +1 #toplam kelime sayısı


#n gram dizileri
#ör: "bugün hava çok güzel" -> input:"bugün hava"; hedef: "çok"
#input: "hava çok"; hedef: "güzel"

input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0] #cümleyi sayısal indekslere çevir
    for i in range(1, len(token_list)):
        n_gram_sequences = token_list[:i +1]
        input_sequences.append(n_gram_sequences)

#en uzun dizinin uzunluğu
max_sequence_lenght = max(len(x) for x in input_sequences)

#padding
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_lenght, padding='pre')

#girdi(x) ve hedef(y)
X=input_sequences[:, :-1] #son kelime hariç tüm kelimeler girdi
y=input_sequences[:, -1] #son kelime 

#one-hot encoding
y=tf.keras.utils.to_categorical(y, num_classes=total_words)

#model kurulumu
model = Sequential();

#embedding katmanı
#total_words = kelime sayısı
#50=embedding boyutu
#input_lenght = giris dizisinin uzunluğu
model.add(Embedding(total_words, 50, input_length=X.shape[1]))

#100=gizli nöron sayısı yani modelin hafıza kapasitesi
model.add(LSTM(100, return_sequences=False))

#output layer
model.add(Dense(total_words, activation="softmax"))

#compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#training
model.fit(X, y, epochs=100, verbose=1)

#metin üretimi
def generate_text(seed_text, next_words):
    """
    seed_text: başlangıç metni
    next_words: kaç kelime üretilecek
    """

    for _ in range(next_words):
        #girdi metnini sayısal verilere dönüştür
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        #padding
        token_list = pad_sequences([token_list], maxlen=max_sequence_lenght-1, padding="pre")

        #modelden olasılık dağılımı al(prediction)
        predicted_probabilities = model.predict(token_list, verbose=0)

        #en yüksek olasılığa sahip kelimenin indexini bul
        predicted_word_index = np.argmax(predicted_probabilities, axis=-1)

        #indexi kelimeye çevir
        predicted_word = tokenizer.index_word[predicted_word_index[0]]

        #kelimeyi metne ekle
        seed_text=seed_text + " " + predicted_word
    return seed_text

seed_text = "bugün hava"
print(generate_text(seed_text, 2))