"""
amaç: 
*bir metin içerisindeki özel varlık isimlerini (kişiler, organizasyonlar, yerler vb.) tespit etmek.
*named entity recognation(NER) işlemini spacy kütüphanesi ile gerçekleştir

adımlar:
*kütüphane import  [python3.12 -m spacy download en_core_web_sm]
*spacy ingilizce dil modelinin yüklenmesi
*metin üzerinde vaarlık ismi tanıma
*bulunan varlıkları terminal'e print ettir
"""

import pandas as pd
import spacy #ner için nlp kütüphanesi

#spacy ingilizce modellerinin yüklenmesi
nlp_model= spacy.load("en_core_web_sm")

#sample metin
sample_text= "Alice works at Amazon and lives in London. She visited the British Museum last weekend."

#metni spacy modeline ver; tokenization, pos tagging, ner otomatik dönüştürsün 
doc= nlp_model(sample_text)

for entity in doc.ents:
    print(entity.text, entity.label_)

#varlıkların lema bilgisi ile birlikte saklanması
entities_list= [(entity.text, entity.label_, entity.lemma_) for entity in doc.ents]

df_entities= pd.DataFrame(entities_list, columns= ["text", "type", "lemma"])
print(df_entities)


