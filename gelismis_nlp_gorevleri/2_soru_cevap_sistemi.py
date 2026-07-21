"""
amaç:
*BERT tabanlı bir soru cevap sistemi kurmak
*model, verilen bir metin(context) içerisinden belirli bir sorunun cevabını bulur.
*Hugging face'in, "bert-large-uncased-whole-word-masking-finetuned-squad" adlı 
önceden eğitilmiş modeli kullanılacak. Bu model SQuAD(stanford question answeing dataset)
üzerinde ince ayar(fine-tuning) yapılmış.

adımlar:
*kütüphenler import
*önceden eğitilen bert modelini ve tokenizer yükle
*predict_answer fon:
    soruyu ve metni tokenize et
    modeli çalıştır
    en yüksek skorlu token aralığını cevap olarak seç
    tokenleri string formuna çevirerek okunabilir hale getir
*test
"""

from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

model_name= "bert-large-uncased-whole-word-masking-finetuned-squad"

tokenizer= BertTokenizer.from_pretrained(model_name) #tokenizer metinleri bert modelinin anlayabileceği sayısal formata dönüştürür

#bert modelinin soru-cevap versiyonu çağır
model= BertForQuestionAnswering.from_pretrained(model_name)

#soru-cevap fonk
def predict_answer(context, question):

    #soru ve metni encode plus ile birleştir
    #return_tensors: çıktıyı pytorch formatına çevirir
    #max_length: bert'in max token kapasitesi
    #truncation: matin 512 tokendan uzunsa keser
    """
    encode_plus çıktıları:

    input_ids:	Her tokenın sayısal kimliği
    attention_mask:	Hangi tokenlar gerçek, hangileri padding? (1 = gerçek, 0 = padding)
    token_type_ids:	Hangi tokenlar soruya, hangileri metne ait? (0 = soru, 1 = metin)
    """
    encoding= tokenizer.encode_plus(question, 
                                    context, 
                                    return_tensors= "pt", 
                                    max_length= 512, 
                                    truncation= True)
    

    #input_ids: her kelimeye karşılık gelen sayısal token kimlikleri
    input_ids= encoding["input_ids"]

    #attention_mask: hangi tokenların dikkate alınacağını belirler
    attention_mask= encoding["attention_mask"]

    #model çalıştır ve tahmin skorlarını al
    #start_score: cevabın başaldığı token olasılıkları
    #end_score: cevabın bittiği token olasılıkları
    with torch.no_grad(): #eğitim yok,backpropagation(güncelleme) yok
        start_scores, end_scores= model(input_ids, attention_mask= attention_mask, return_dict= False)

    #en yüksek olasılıklı başlangıç ve bitiş token indeksleri al
    # .item() ile PyTorch tensoründen Python int'ine çevirir.
    start_index= torch.argmax(start_scores, dim=1).item()
    end_index= torch.argmax(end_scores, dim=1).item()

    #input_ids üzerinden cevaba denk gelen token aralığını al
    answer_tokens= tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index +1]) #slicing [:] bitiş indeksini dahil etmez. end_index'i de almak için +1 eklenir.

    #tokenları birleştirerek okunabilir yazı haline getir
    answer= tokenizer.convert_tokens_to_string(answer_tokens)

    return answer

question= "what is the capital of france"
context= "france, officially the french republic, is a country where capital is paris"
answer= predict_answer(context, question)
print(f"answer: {answer}")

