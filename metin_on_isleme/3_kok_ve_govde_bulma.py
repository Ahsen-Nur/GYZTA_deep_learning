"""
* kok bulma -> stemming: porter stemmer
* govde bulma -> lemmatization: word net lemmatizer
"""

import nltk
nltk.download("wordnet")  #lemma bulmak için gerekli wordnet veri tabanı
nltk.download("omw-1.4")  #wordnet için ek dil desteği


#stemming
from nltk.stem import PorterStemmer     #ingilizce için stemmer algoritması
stemmer = PorterStemmer()   #porter stemmer nesnesi oluşturma
word_stem = ["playing", "played", "plays", "happier", "happiest", "running", "runner"]

stems = [stemmer.stem(word) for word in word_stem]
print(f"original kelimeler: {word_stem}")
print(f"Stemming sonucu: {stems}")


#lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()   #lemmatizer nesnesi oluşturma
word_lemma = ["running", "ran", "gone", "better", "children"]

lemmas = [lemmatizer.lemmatize(word) for word in word_lemma]
print(f"original kelimeler: {word_lemma}")
print(f"Lemmatization sonucu: {lemmas}")