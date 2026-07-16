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

from nltk.sentiment.vader import SentimentIntensityAnalyzer #duygu analizi için
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report

nltk.download("vader_lexicon") #vader sözlüğü
nltk.download("stopwords")
nltk.download("wordnet") #wordnet sözlüğü, lemmatizer için
nltk.download("own-1.4") #wordnet'in çoklu dil desteği
nltk.download("punkt_tab") #tokenization için 


reviews_df= pd.read_csv("amazon.csv")
print(reviews_df.head())

#metin temizleme ve ön işleme
lemmatizer= WordNetLemmatizer()

def clean_and_preprocess(text):
    tokens= word_tokenize(text.lower())
    filtered_tokens= [token for token in tokens if token not in stopwords.words("english")]
    lemmatized_tokens= [lemmatizer.lemmatize(token) for token in filtered_tokens]

    #kelimeleri birleştir
    processed_text= " ".join(lemmatized_tokens)

    return processed_text

#tüm yorumlara uygula
reviews_df["cleaned_review"]= reviews_df["reviewText"].apply(clean_and_preprocess)
print(reviews_df.head())


#sentiment analysis function
analyzer= SentimentIntensityAnalyzer()

def analyze_sentiment(text):

    #vader puanlarını al
    score= analyzer.polarity_scores(text)

    sentiment= 1 if score["pos"] > 0 else 0
    return sentiment


reviews_df["predicted_sentiment"]= reviews_df["cleaned_review"].apply(analyze_sentiment)
print(reviews_df.head())

#model değerlendirme
cm= confusion_matrix(reviews_df["Positive"], reviews_df["predicted_sentiment"])
print(f"confusion matrix: \n{cm}")

cr= classification_report(reviews_df["Positive"], reviews_df["predicted_sentiment"])
print(cr)





