"""
Amaç:
    *fazla boşlukları temizleme
    *noktalama işaretlerini temizleme
    *küçük harfe çevirme
    *özel karakterleri temizleme
    *yazım hatalarını düzeltme
    *html etiketlerini temizleme

pip install textblob beautifulsoup4
"""

# Fazla boşlukları temizleme
raw_text = "Python,      Google    NLP         dersi."
print(raw_text.split())
normalized_text_1 = " ".join(raw_text.split())
print(f"Fazla boşlukları temizlenmiş metin: {normalized_text_1}")


#büyük küçük harf dönüşümü
raw_text = "PYTHON, GooGle NLP"
normalized_text_2 = raw_text.lower()
print(f"Küçük harfe çevrilmiş metin: {normalized_text_2}")


#noktalama işaretlerini temizleme
import string
raw_text = "AI Natural-Language-Processing!"
normalized_text_3 = raw_text.translate(str.maketrans(" ", " ", string.punctuation))
print(f"Noktalama işaretleri temizlenmiş metin: {normalized_text_3}")


#özel karakterleri temizleme
import re
raw_text = "Natural @ Languagae % Processing"
normalized_text_4 = re.sub(r"[^a-zA-Z0-9\s]", "", raw_text)
print(f"Özel karakterler temizlenmiş metin: {normalized_text_4}")


#yazım hatalarını düzeltme
from textblob import TextBlob
raw_text = "I havv a speling eror."
normalized_text_5 = TextBlob(raw_text).correct()
print(f"Yazım hataları düzeltilmiş metin: {normalized_text_5}")


#html etiketlerini temizleme
from bs4 import BeautifulSoup
raw_html = "<div> 2045 Google </div>"
normalized_text_6 = BeautifulSoup(raw_html, "html.parser").get_text()
print(f"HTML etiketleri temizlenmiş metin: {normalized_text_6}")