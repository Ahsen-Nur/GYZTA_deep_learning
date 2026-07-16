"""
amaç:
*spacy kullanarak verilen bir cümlenin her kelimesini(token) incelemek
*her kelimenin kök(lemma), pos(dilbilgisel kategorisi), morfoloji, çoğul olup olmadığı 
gibi bilgilerin çıkarılması

adımlar:
*spacy ingilizce modeli yükle
*örnek bir cümle üzerinden nlp işlemi uygulama
*her kelimenin özelliklerinni ekrana yazdır.

"""

import spacy #nlp işlemleri için

nlp_model= spacy.load("en_core_web_sm")

sentence= "They are palying football in the parks."

#nlp işleminden geçer; tokenization, pos tagging, dependency parse, lemma, morphology
doc= nlp_model(sentence)

for token in doc:
    print(f"text: {token.text}") #kelimenin kendisi
    print(f"lemma: {token.lemma_}") #kök hali
    print(f"pos: {token.pos_}") #genel dil bilgisel kategorisi
    print(f"tag: {token.tag_}") #daha detaylı pos etiketi
    print(f"dependency: {token.dep_}") #cümledeki sözdizimselrolü (ör: özne, yüklem)
    print(f"shape: {token.shape_}") #kelimenin karakter yapısı(Xxx, xxx "büyük/küçük harf vs")
    print(f"is alpha: {token.is_alpha}")
    print(f"is stop: {token.is_stop}")
    print(f"morphology: {token.morph}")
    print(f"is plural: {"Number=Plur" in token.morph}") #çoğul mu
    print("-"*50)

    







