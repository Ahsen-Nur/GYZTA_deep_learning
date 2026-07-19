"""
amaç: 
*hugging face'in transformers kütüphanesini kullanarak bir metin özetleme pipeline'ı 
ve verilen uzun metni kısa bir özet haline getirmek

adımlar:
*gerekli kütüphanleri import etme
*özetleme summarization pipeline'ı yükleme
*uzun bir metin tanımlama
*modeli çalıştırarak özet oluşturma
*sonucu ekrana yazdırma
"""


from transformers import pipeline #hugging face pipeline ile önceden eğitilmiş modeller kullanılabilir

#özetleme pipeline yükle
summarizer= pipeline("summarization") #summarization parametresi: modele özetleme yaptırır, arkada llm modeli çağırır
print(summarizer.model.config._name_or_path)
#print(summarizer.model)

text= """
The Scent of Lost Letters
Near the crowded avenues of Güngören, time slowed down at Mr. Rıza’s antiquarian bookstore, "Beyond Time." The shop smelled of old paper and sage tea, rescuing visitors from the stressful outside world. Mr. Rıza, a wise man in his seventies, had dedicated his life to preserving words.
One morning, an architecture student named Elif entered. She was researching the lost historical fabric of Istanbul for her graduation project. Instead of technical drawings, she was looking for the soul of the people who once lived in those old houses. Understanding her intent, Mr. Rıza handed her an old, delicate diary with a slightly rusted lock and the initials "L.S." engraved in gold. 
Opening it, Elif was transported to the 1940s. The diary belonged to Leyla, a young woman living in a wooden mansion in Büyükada. Leyla wrote passionately about architecture, expressing deep sadness as Istanbul's traditional bay-windowed houses slowly succumbed to concrete. 
As Elif read on, the tone shifted to an impossible love story. Leyla had fallen in love with a marine engineer from Galata, a match their families disapproved of. Secret letters sent between Galata Tower and Büyükada were their only connection. When World War II forced the young man to sail away, Leyla resisted her family's pressure to marry someone else. Near the end, Leyla wrote: "Istanbul is not just stone and soil; it is made of the loves and separations echoing in its streets."
Tears in her eyes, Elif realized how to complete her project. She needed to tell the real stories and human lives inside the buildings, proving that history was still breathing. Weeks later, Elif amazed her university jury by combining her technical architectural drawings with Leyla’s moving story. From that day on, walking through Istanbul, Elif always heard Leyla’s elegant words whispered in the wind.
"""

#model çalışır, özetleme gerçekleşir
summary= summarizer(
    text,
    max_length=80, #özet max 20 token olur
    min_length=5,  #özet min 5 token olur
    do_sample= True #rastgelelik ekleyerek modelin her seferinde farklı özetler üretmesini sağlar
)

#sonuç yazdır
#summarizer fonk. bir liste return eder, her öge bir sözlük yapısındadır
print(summary[0]["summary_text"])




