"""
amaç:
*bu uygulama bert modeli kullanrak metin benzerliği(semantic similarity) analizi gerçekleştirecek.
*bir sorgu cümlesi (query) ile bir dizi belgenin(documents) anlamca ne kadar benzer olduğu ölçülür.
*her metin bert modelinden elde edilen embedding(vektör temsili) ile temsil edilir
*benzerlik ölçümü için kosinüs benzerliği(cosine similarity) kullanılır.

adımlar:
*kütüphane import
*bert model ve tokenizer yükle
*örnek belgeleri ve sorgu cümlesini oluştur
*her metni embedding(vektör) haline getir
*sorgu ve belgeler arasında benzerliği hesapla
*en benzer belgeyi belirle ve yazdır

"""

from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name= "bert-base-uncased" 
tokenizer= BertTokenizer.from_pretrained(model_name)
model= BertModel.from_pretrained(model_name)

#veri, belge oluşturma
documents= [
    "machine learning is as field of artificial intelligence",
    "natural language preprocessing involves understanding human language",
    "artificial intelligence encompasses machine learning and natural language processing(nlp)",
    "deep learning is a subset of machine learning",
    "data science combines statistics, data analysis and machine learning"
]

#kullanıcıların sorgusu
query= "what is deep learning?"

def get_embedding(text):
    """
    verilen bir metni BERT kullanarak sayısal vektörlere dönüştür:
    tokenization yapılır
    model çalıştırılır
    embedding yapılır
    """
    #return tensors: pytorch formatında tensörlere çevirme
    inputs= tokenizer(text, return_tensors="pt", truncation= True, padding=True)

    #model çalıştır
    #inputs bir sözlük içerir: {'input_ids': ..., 'attention_mask': ..., ...}. 
    # **inputs bu anahtar-değer çiftlerini model(input_ids=..., attention_mask=...) 
    # şeklinde fonksiyon argümanı olarak açar.
    outputs= model(**inputs)

    #son gizli katmandan ortalama alınarak tek vektör elde edilir
    #BERT her token için ayrı vektör verir. Ama gereken cümlenin tek bir vektörü. 
    # Bunun yollarından biri de mean pooling. tüm token vektörlerinin ortalamasını alır.
    # Avantajı tüm kelimelerin katkısı eşittir ancak dezavantajı cümledeki önemsiz kelimeler de aynı ağırlıktadır.
    #diğer yöntemler: [CLS] tokeni, max pooling
    last_hidden_state= outputs.last_hidden_state
    embedding= last_hidden_state.mean(dim=1)

    #PyTorch tensorleri computational graph (hesaplama grafiği) içindedir. 
    # Yani model eğitilirken gradyanlar bu tensordan geriye doğru hesaplanabilir. 
    # Burada sadece tahmin (inference) yapılıyor; gradyan takibine ihtiyaç yok. 
    # detach() bu bağlantıyı koparır.
    return embedding.detach().numpy() #cosine_similarity fonksiyonu sklearn'ın bir parçasıdır ve numpy array bekler. PyTorch tensorünü numpy'a çevrilir.


#belgeler ve sorgu için embedding hesapla, embedding matisi oluşturma
#vstack (Vertical stack) = dikey istifleme
doc_embeddings= np.vstack([get_embedding(doc) for doc in documents]) #her belge için embedding
query_embedding= get_embedding(query)

#benzerlik hesapla: sonuç=1 ise tamamen benzer; sonuç=0 ise alakasız
#Euclidean mesafe, vektörün uzunluğunu da hesaba katar. Uzun metinler doğal olarak 
# daha büyük norma sahip olabilir. Cosine sadece yönü ölçer.
similarities= cosine_similarity(query_embedding, doc_embeddings)

#sonuç yazdır
for i, score in enumerate(similarities[0]):
    print(f"document: {documents[i]} \n similarity score: {score:.4f}\n")

#en yüksek benzerliğe sahip belgeyi bul
most_similar_index= similarities.argmax()
print(f"most similar document: {documents[most_similar_index]}")











