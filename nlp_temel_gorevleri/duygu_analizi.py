"""
amaç: 
*amazon yorumlarını pozitif/negatif sınıflandırma
*binary classification problemi

adımlar:
*veri seti yükleme
*metin temizleme ve ön işleme(tokenization, stopwords, lemmatization)
*duygu analizi (sentiment intensity analizer - VADER)
*tahmin sonuçlarını confusion matrix ve classification report ile değerlendir
"""

import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer #VADER duygu analizi aracı. Metni -1 ile +1 arasında puanlar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report

nltk.download("vader_lexicon") #VADER'ın duygu sözlüğü. Kelimelerin pozitif/negatif ağırlıklarını içerir
nltk.download("stopwords")
nltk.download("wordnet") #İngilizce sözlük ağı. Lemmatizer kelimelerin kökünü buradan bulur
nltk.download("own-1.4") #wordnet'in çoklu dil desteği
nltk.download("punkt_tab") #Cümleleri kelimelere ayırmak için gerekli tokenization modeli 


reviews_df= pd.read_csv("amazon.csv")
print(reviews_df.head())

#metin temizleme ve ön işleme
lemmatizer= WordNetLemmatizer()

def clean_and_preprocess(text):
    tokens= word_tokenize(text.lower()) #word_tokenize(...): Cümleyi kelimelere ve noktalama işaretlerine ayırır.
    filtered_tokens= [token for token in tokens if token not in stopwords.words("english")] #Bu kelimeler duygu taşımaz. Her cümlede olduğu için modelin (veya VADER'ın) ayırt ediciliğini düşürürler.
    lemmatized_tokens= [lemmatizer.lemmatize(token) for token in filtered_tokens] #Kelimenin sözlük köküne (lemma) indirger. WordNetLemmatizer, kelimenin türüne (isim, fiil, sıfat) göre kökünü bulur.

    #kelimeleri birleştir
    processed_text= " ".join(lemmatized_tokens)

    return processed_text

#tüm yorumlara uygula
reviews_df["cleaned_review"]= reviews_df["reviewText"].apply(clean_and_preprocess)
print(reviews_df.head())


#sentiment analysis function
#VADER, rule-based bir duygu analiz aracıdır. Makine öğrenmesi yapmaz; 
#önceden hazırlanmış bir sözlük kullanır. Özellikle sosyal medya metinleri 
#için optimize edilmiştir: Büyük harfleri dikkate alır, Emojileri puanlar,
#Noktalama işaretlerini dikkate alır, Kelimeyi bağlama göre ayarlar,
analyzer= SentimentIntensityAnalyzer()

def analyze_sentiment(text):

    #vader puanlarını al
    #polarity_scores(text) ne döndürür?
    """
{
    'neg': 0.0,    # Negatif duygu oranı (0-1)
    'neu': 0.294,  # Nötr duygu oranı (0-1)
    'pos': 0.706,  # Pozitif duygu oranı (0-1)
    'compound': 0.7096  # Bileşik skor (-1 ile +1 arası)
}
    """
    #Bu skorların hesaplanması şöyledir; VADER sözlüğünde her kelimenin bir duygu ağırlığı vardır. 
    #Metindeki kelimelerin ağırlıkları toplanır, bağlam kuralları (inkar, büyütme, vb.) 
    #uygulanır ve normalize edilir.
    score= analyzer.polarity_scores(text) 

    sentiment= 1 if score["pos"] > 0 else 0
    return sentiment

#Her temizlenmiş yorum için VADER tahmini yapılır ve yeni sütuna kaydedilir.
reviews_df["predicted_sentiment"]= reviews_df["cleaned_review"].apply(analyze_sentiment)
print(reviews_df.head())

#model değerlendirme
cm= confusion_matrix(reviews_df["Positive"], reviews_df["predicted_sentiment"])
print(f"confusion matrix: \n{cm}")


# Precision =  TP / (TP + FP)  -> Pozitif dediklerimizin ne kadarı gerçekten pozitif?
# Recall = TP / (TP + FN) -> Gerçek pozitiflerin ne kadarını yakaladık?         
# F1-Score = 2 × (Precision × Recall) / (Precision + Recall) -> Precision ve Recall'in harmonik ortalaması          
# Accuracy = (TP + TN) / Toplam -> Genel doğruluk oranı                                

cr= classification_report(reviews_df["Positive"], reviews_df["predicted_sentiment"])
print(cr)





