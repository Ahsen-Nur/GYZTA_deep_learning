"""
problem tanımı: AI Proje Yöneticisi
    *bu proje, bir proje dokümanını (pdf) okuyarak ekip üyelerine gerçek zamanlı olarak 
    görev hatırlatmaları yapan bir ai sistemi oluşturulacak yani ai agent

    *yapay zeka yöneticisi:
        *bir proje planı oluşturulur (ne yapılacak, takvim, ekip vs) ve ai yöneticisine verilir
        *bu dosya/dosyalarda bulunan görev zamanına göre ai yöneticisi ekip üyelerine taskları sorar
        *çalışan doğal dilde cevap verir ve ai yöneticisi bunu analiz eder
        *eğer görev tamamlandıysa (ai yöneticisinin bunu anlaması lazım) devam, tamamlamadıysa bir daha sorar
        *tüm sorular ai yöneticisi tarafından ekip üyelerinin geçmiş cevaplarına göre kişiselleştirilerek 
        tekrar sorulur veya yeni soru sorulur
        
    *simülasyon ortamı: 
        10 sn'de bir 1 dk ilerleyen simülasyon saati

*veriseti:
    *proje ile ilgili dokümanlar:
        teknik şartname, proje takvimi, proje sözleşmesi ve ekleri, 
        literatür taraması, yazılım gereksinim özellikleri, yazılım tasarım tanımı, 
        yazılıım test tanımı, fabrika kabul testleri, müşteri kabul testleri...

araçlar ve teknolojiler:
    *gemini, rich(terminalde renkli ve biçimli çıktı)

plan/program:
    *proje dokümanı oluşturma ve sonrasında pdf reader
    *gemini agent:
        *taskların sorulması
        *taskların tamamlanıp tamamlanmadığının anlaşılması
    *simülasyon ile parçaların birleştirilmesi

pip install google-generativeai python-dotenv rich PyPDF2
"""

import time #zaman simülasyonu için
from datetime import datetime, timedelta #tarih ve zaman işlemleri için modüler bir yapı oluşturur
from pdf_reader import extract_tasks_from_pdf
from rich import print #zengin ve renkli terminal çıktısı için
from gemini_agent import is_task_completed, generate_followup_questions


#taskları hafızaya gönder
task_memory= {}

#simülasyon çalıştırma 
def run_scheduler(pdf_path= "Rapor.pdf", delay_sec= 10):

    #belirtilen pdf dosyasından görevleri çıkart
    tasks= extract_tasks_from_pdf(pdf_path)

    #simülasyon başlangıç zamanı tanımla
    sim_time= datetime(2025, 8, 25, 11, 59) #25.08.2025 11:59

    #simülasyon başlangıcını kullanıcıya bildir
    print(f"[bold green] simülasyon başladı [/bold green] -> başlangıç: {sim_time.strftime('%d.%m.%Y %H:%M')}")

    #simülasyon döngüsü, 1 dk'lık süreyi 10 saniyede ilerleten döngü başlangıcı
    while True:
        sim_time += timedelta(minutes=1) #simülasyon zamanını 1 dk ilerletir
        sim_time_str= sim_time.strftime("%d.%m.%Y %H:%M")
        print(f"\n[bold white on black] simülasyon saati: {sim_time_str}[/bold white on black]")

        #her görev için kontrol yapılır
        for task in tasks:

            ts= task["timestamp"] #görevin hedef zamanı
            kisi= task["person"]
            gorev= task["task"]
            key= f"{ts}_{kisi}" #aynı görev ve kişiyi unique olarak tanımlamak için bir anahtar tanımlanır

            if ts <= sim_time:
                #daha önce verilen cevapları bellekten al
                onceki_cevaplar= task_memory.get(key, [])

                #önceki cevaplar varsa görevin tamamlanıp tamamlanmadığını sorgula
                if onceki_cevaplar:
                    tamam_durumu= is_task_completed(kisi, gorev, onceki_cevaplar, sim_time_str)
                    if tamam_durumu == "tamamlandı":
                        continue #görev tamamlandı, soru sorma, başkasına geç
                    else:
                        print(f"[yellow]{kisi} görevini henüz tamamlamadı. Tekrar soruyor...[/yellow]")

                #gemini ile soru sorma
                soru= generate_followup_questions(
                    person= kisi,
                    task= gorev,
                    current_time= sim_time_str,
                    previous_responses= onceki_cevaplar
                )

                print(f"[bold red]{kisi}[/bold red] kişisine AI yöneticisi tarafından oluşturulan soru:")
                print(f"[bold blue]{soru}[/bold blue]")

                cevap= input("cevap: ").strip() #ekip üyesi cevabı

                task_memory.setdefault(key, []).append({
                    "time": sim_time.strftime("%H:%M"),
                    "response": cevap
                })

        time.sleep(delay_sec)

if __name__ == "__main__":
    run_scheduler()

