"""
amaç:
*bir cümledeki her kelimenin dilbilgisel türünü bulmak
*spacy kullanarak her token için pos etiketi bulalım

adımlar:
*spacy ingilizce modeli yükle
*örnek cümle oluştur ve nlp modelinden geçir
*her kelimenin pos etiketini yazdır
"""

import spacy

nlp_model= spacy.load("en_core_web_sm")

sentence= "Can you recommend a good restaurant in London."

doc= nlp_model(sentence) #pos tagging

for token in doc:
    print(f"{token.text:12} {token.pos_}")
