"""
amaç:
*kullanıcı-ürün puanlarından öğrenerek kullanıcıların ürünlere vereceği 
puanı tahmin eden bir sistem
*yaklaşım: neural collaborative filtering en temel hali matrix factorization
benzeri

adımlar:
*kütüphane import
*örnek veri kümesi oluştur(user_id, item_id, rating)
*train-test
*embedding tabanlı bir neural network model tanımla
*model eğitimi
*test
*örnek kullanıcı-ürün çifti üzerinden puan tahmini
"""


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)


#örnek veri kümesi(user_id, item_id, rating)
user_ids_all= np.array([0,1,2,3,4,0,1,2,3,4], dtype="int32")
item_ids_all= np.array([0,1,2,3,4,1,2,3,4,5], dtype="int32")
ratings_all= np.array([5,4,3,2,1,4,5,3,2,1], dtype="float32")

#Embedding katmanı input_dim olarak toplam sayıyı ister.
num_users= int(user_ids_all.max()) +1
num_items= int(item_ids_all.max()) +1

#train_test_split
user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(
    user_ids_all, item_ids_all, ratings_all, test_size=0.2, random_state=42)


#model oluşturma/ klasik Matrix Factorization'ın sinir ağı ile yapılmıştır
def create_recommender_model(n_users, n_items, embedding_dim=8, lr=0.01):
    """
    embedding_dim= embedding vektör boyutu
    lr= learning rate
    """

    #input katmanı
    user_input= Input(shape(1,), name= "user_input")
    item_input= Input(shape(1,), name= "item_input") 

    #embedding katmanı
    user_embedding= Embedding(input_dim= n_users, output_dim= embedding_dim, name= "user_embedding")(user_input)
    item_embedding= Embedding(input_dim= n_items, output_dim= embedding_dim, name= "item_embedding")(item_input)

    #vektörleri düzleştir
    user_vec= Flatten(name= "user_vec")(user_embedding)
    item_vec= Flatten(name= "item_vec")(item_embedding)

    #tahmin temeli
    #İki vektör ne kadar paralelse (benzer yönde ise), iç çarpım o kadar büyük olur. 
    #Yani kullanıcı ve ürün uyumluysa skor yüksek çıkar. Bu skor, tahmini puandır.
    dot_score= Dot(axes=1, name= "dot_user_item")([user_vec, item_vec])

    #Dot product'tan gelen skalere (tek sayı) bir lineer dönüşüm uygular
    output= Dense(1, activation="linear", name="rating")(dot_score)

    #modeli derle
    model= Model(inputs= [user_input, item_input], outputs=output, name= "EmbeddingDotRecommender")
    model.compile(optimizer= Adam(learning_rate= lr), loss= "mean_squared_error", metrics=["mae"])

    return model

#model oluştur ve eğit
#Input: user_ids_train ve item_ids_train aynı anda verilir.
#Embedding: Her ID vektöre çevrilir.
#Dot Product: Uyum skoru hesaplanır.
#Dense: Tahmini puan üretilir.
#Loss: Gerçek puanla karşılaştırılır.
#Backpropagation: Hata geriye yayılır, embedding vektörleri güncellenir.

embedding_dim=8
learning_rate=0.01
model=create_recommender_model(num_users, num_items, embedding_dim=embedding_dim, lr=learning_rate)

history= model.fit(
    [user_ids_train, item_ids_train],
    ratings_train,
    epochs=100,
    batch_size=4,
    verbose=1,
    validation_split=0.1
)

#test
test_loss, test_mae= model.evaluate([user_ids_test, item_ids_test], ratings_test, verbose= 0)
print(f"test_loss: {test_loss}, test_mae: {test_mae}")








