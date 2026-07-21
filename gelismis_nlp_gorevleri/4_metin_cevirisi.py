"""
amaç:
* uygulamanın amacı, Hugging face in MarianMT(marian machine translation) modelini kullanarak çeviri yapmak
*sequence-to-sequence (seq2seq) mimarisine dayalı bir nöral makine çevirisi (NMT) sistemi 
*ingilizce -> fransızca
*model Helsinki NLP grubu tarafından geliştirilmiş
*bu modeller çok sayıda dil çifti arasında çeviri yapmak üzere eğitildi

adımlar:
*kütüphane import
*model adı belirleme
*tokenizer ve model yükleme
*girdi metni tanımla
*tokenizer ile metni modelin anlayabileceği formata çevir
*model ile çeviri işlemi
*token id'leri tekrar metne dönüştür ve sonucu ekrana yazdır.
"""

from transformers import MarianMTModel, MarianTokenizer

model_name= "Helsinki-NLP/opus-mt-en-fr"

#tokenizer: metni sayısal tokenlara dönüştürür
tokenizer= MarianTokenizer.from_pretrained(model_name)

#model tokenları alır ve hedef dile çevrilmiş tokenlar üretir
model= MarianMTModel.from_pretrained(model_name)

#girdi metni
text= "hello, what is your name"

#tokenization 
inputs= tokenizer(text, return_tensors= "pt", padding= True)

#model ile çeviri yap
translated_tokens= model.generate(**inputs) #çeviri çıktısını üret

#çıktıyı okuanbilir metne çevir
#skip special tokens ile özel semboller kaldırılır. <pad>, <eos>
#Encoder kaynak dili anlar, decoder hedef dili üretir.
translated_text= tokenizer.decode(translated_tokens[0], skip_special_tokens= True)

print(f"translated text: {translated_text}")


"""
Encoder, cümleyi kelime kelime okur ama asıl yaptığı iş, her kelimenin cümle içindeki 
bağlamını anlayarak bir "hafıza bankası" oluşturmaktır.

    * encoder' a giren input_ids; çıkan ise  her token için 512 boyutlu (varsayılan) bir vektör

Encoder işini bitirdikten sonra kelimeler unutulmaz. Decoder, bu vektörlere 
bakarak sorgulama yapar.

Decoder'ın en kritik özelliği: Tek seferde tüm cümleyi üretemez. Sadece bir sonraki 
kelimeyi tahmin edebilir. O yüzden "otoregresif" (autoregressive) denir: kendi önceki 
çıktısı, sonraki girdisi olur.

Decoder'ın kendi içinde bir hafızası(state) vardır ama bu hafıza, bir sonraki tokenı 
üretmek için yeterli değildir.  Onu dışarıdan, girdi olarak tekrar beslememiz gerekir. 
Her adımda bu hafıza güncellenir. 
Başlangıçta boştur (veya sadece başlangıç tokenı vardır).

    *Decoder'a verilen ilk şey başlangıç tokenı <s> (start) olur. 
    *Decoder'ın içinde iki önemli mekanizma çalışır:
        a) Self-Attention (Kendi Ürettiklerine Bakma): Decoder, şimdiye kadar ürettiği tokenlara bakar. 
        Şu an sadece <s> var, o yüzden sadece kendine bakar.

        b) Cross-Attention (Encoder'ın Hafızasına Bakma): Decoder, Encoder'ın ürettiği bağlam vektörlerine
        sorgu (query) gönderir: 
            "Ben şu an <s> tokenıyım. Encoder'daki hangi İngilizce kelimeyle ilgiliyim?"
        Cross-attention mekanizması, "hello" ve "what" vektörlerine yüksek ağırlık verir. 
        Çünkü <s> genel bir başlangıçtır ve tüm cümleye bakması gerekir.

    *Cross-attention sonucunda Decoder, cümlenin genel anlamını kavrar. Şimdi bir 
    sonraki tokenı tahmin eder.

"""