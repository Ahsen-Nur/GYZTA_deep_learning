"""
amaç:
*aynı kelimenin farklı bağlamlarda farklı anlamlarını bulmak
*bunu yaparken lesk algoritması kullanılacak

yöntem: lesk algoritması
*lesk, bir kelimenin doğru anlamını belirlemek için bağlamdaki(cümledeki) diğer 
kelimeler ile karşılaştırma yapar
*WordNet sözlüğündeki tanımlar ile cümledeki kelimeler arasındaki ortak kelimeleri sayar
*en çok örtüşen anlam o kelimenin doğru anlamı olarak seçilir.
*ör: "bank"= para yatırma | "bank"= nehir kıyısı(river bank)

adımlar:
*gerekli nltk paketleri indir
*ilk cümle üzerinde bnak kelimesini çözümle
*ikinci cümle üzerinde bank kelimesini çözümle
*her cümle için bulunan anlamı ekrana yazdır
"""


import nltk
from nltk.wsd import lesk

nltk.download("wordnet") #wordnet sözlüğü
nltk.download("own-1.4") #wordnet'in çoklu dil desteği
nltk.download("punkt_tab") #tokenization için gerekli


#ilk cümle örneği
sentence1= "I go to the bank to deposit money"
target_word1= "bank"

sense1= lesk(nltk.word_tokenize(sentence1), target_word1)
print(f"sentence: {sentence1}")
print(f"word: {target_word1}")
print(f"predicted sense: {sense1.definition()}")


sentence2= "The river bank is flooded after the heavy rain."
target_word2= "bank"

sense2= lesk(nltk.word_tokenize(sentence2), target_word2)
print(f"sentence: {sentence2}")
print(f"word: {target_word2}")
print(f"predicted sense: {sense2.definition()}")