"""
Amaç: IMDB veri seti üzerinden GRU tabanlı bir duygu analizi modeli

*kütüphane kurulumu
*padding
*GRU tabanlı model kurulması
*modelin derlenmesi ve eğitilmesi
*evaluation
*yeni yorum tahmini için fonk.
"""

import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense


#imdb veri setinde en sık geçen 10000 kelime kullaılsın
num_words=10000 #sözlükte tutulacak kelime sayısı
max_sequence_length=200 #her yorum 200 kelimeden oluşacak

#X_train ve X_test = yorumlar
#y_train ve y_test = etiketler
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_words)
print(f"Train boyutu: {len(X_train)}, test boyutu: {len(X_test)}")

#padding
#padding='pre' (baştan doldur), truncating='pre' (baştan kes)
#Dil modelleri genellikle cümlenin sonunu daha iyi hatırlar. Baştan kesmek/eklemek, son kelimelerin model tarafından görülmesini garanti eder.
X_train_padded = pad_sequences(X_train, maxlen = max_sequence_length, padding='pre', truncating='pre')
X_test_padded = pad_sequences(X_test, maxlen = max_sequence_length, padding='pre', truncating='pre')

print(f"X_train şekli:{X_train_padded.shape}")
print(f"X_test şekli:{X_test_padded.shape}")


#model oluşturma

#embedding layer
#Bu katman, (25000, 200) boyutundaki girdiyi (25000, 200, 100) boyutuna çevirir.
#Her 0-9999 arası tam sayı, 100 boyutlu bir öğrenilebilir vektör ile değiştirilir. 
#Bu vektörler modelin eğitilebilir parametreleridir; başlangıçta rastgele atanır ve 
# geri yayılım (backpropagation) ile güncellenir.
embedding_dim=100 #kelimeler 100 boyutlu vektörler ile temsil edilsin

model = Sequential()

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_dim,
                    input_length=max_sequence_length
                    ))


#gru layer
model.add(GRU(units=64,
              return_sequences=False))

#output layer
model.add(Dense(1, activation="sigmoid"))


#model compiler
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print(model.summary())


#model training
history = model.fit(X_train_padded,
                    y_train,
                    epochs=3,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)


#model evaluate
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"test loss: {loss:4f}")
print(f"test accuracy: {accuracy:4f}")


#user test
#imdb dataset sayısal index kullanmadığı için doğrudan kelimeler ile test edilmez bu yüzden
#word_index alıp index_to_word mapping yapılması lazım

word_index = imdb.get_word_index() #kelime->index sözlüğü

#indexleri kelimeye çevirmek için test mapping
index_to_word = {v+3: k for k, v in word_index.items()}
index_to_word[0]= "<PAD>" #padding
index_to_word[1]= "<START>" #cümlenin başlangıcı
index_to_word[2]= "<>UNK" #bilinmeyen cümle

def decode_review(encoded_reviews):
    """
    sayısal bşr yorumu tekrar kelimelere çevirir
    """
    return " ".join([index_to_word.get(i, "?") for i in encoded_reviews])


def classify_review(review_sequence):
    """
    sayısal imdb yorumunu sınıflandırır
    """
    padded = pad_sequences([review_sequence], maxlen = max_sequence_length)
    prob = model.predict(padded)[0][0]
    label = "positive" if prob > 0.5 else "negative"
    return label, prob


decoded =decode_review(X_test[0])
print(decoded)
pred_label, prob = classify_review(X_test[0])
print(f"tahmin: {pred_label}, olasılık: {prob}")