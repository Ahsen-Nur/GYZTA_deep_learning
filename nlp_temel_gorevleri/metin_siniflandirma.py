"""
amaç:
*sms spam collection dataset üzerinden "spam" ve "ham(normal mesaj)" sınıflandırma
*bu problem bir binary classification problemi decision tree algoritması ile çözülecek

adım:
*gerekli küütphaneleri import et
*veri seti yükleme
*metin ön işleme(temizlik + lower + stopwords + lemmatization)
*feature extraction(metin temsili): bag of words(countVectorizer)
*ml model eğitimi
*evaluation(test)/başarı metrikleri: confusion matrix + accuracy
"""

import pandas as pd #veri okuma işleme
import nltk #metin işleme
import re #regex ile metin temizleme

from nltk.corpus import stopwords #anlamsız kelimeleri çıkarma
from nltk.stem import WordNetLemmatizer #lemma bulma işlemi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #bag of words
from sklearn.tree import DecisionTreeClassifier #sınıflandırıcı model
from sklearn.metrics import confusion_matrix


#veriseti yükleme
df=pd.read_csv("sms_spam.csv", encoding="latin-1")
print(df.head())

#kolon isimleri değiştirme
df.columns = ["label", "text"]

#eksik değer kontrol
print(f"eksik değer sayıları: \n {df.isna().sum()}")


#preprocessing(temizlik, lower, stopwords, lemmatization)
nltk.download("stopwords") #stopwords listesini indirir
nltk.download("wordnet") #lemmatizer için wordnet indirir
nltk.download("omw-1.4")

#lemmatizer oluştur
lemmatizer=WordNetLemmatizer()

#temizlenmiş verilerin listesi
clean_texts = []

for msg in df["text"]:

    temp=re.sub("[^a-zA-Z]", " ", msg) #harf olmayan karakterleri çıkart
    temp= temp.lower() #tüm harfleri küçük hale getir
    temp=temp.split() #kelimeleri ayır

    #stopwords çıkart
    temp= [word for word in temp if word not in stopwords.words("english")]

    #lemma
    temp= [lemmatizer.lemmatize(word) for word in temp]

    #kelimeleri tekrar birleştir
    temp= " ".join(temp)

    clean_texts.append(temp)


df["clean_text"] = clean_texts
print(df.head())


#eğitim ve test veri seti
X= df["clean_text"] #bağımsız değişkenler
y= df["label"] #hedef değişken (spam/ham)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


#feature extraction (bow)
cv=CountVectorizer()
X_train_cv = cv.fit_transform(X_train) #eğitim verisini dönüştür
X_test_cv = cv.transform(X_test) #test veri setini dönüştür

#model eğitimi
dt_model= DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cv, y_train) #training

#model değerlendirme 
y_pred= dt_model.predict(X_test_cv) #test veri seti üzerinden tahmin yap

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#accuracy
accuracy= 100*(conf_matrix.trace()/conf_matrix.sum())
print(accuracy)