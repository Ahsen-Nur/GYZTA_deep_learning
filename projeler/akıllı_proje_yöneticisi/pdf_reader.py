from PyPDF2 import PdfReader #pdf dosyasını okumak ve içeriğini çıkartmak
import re #düzenli ifadeler ile metin içerisinden desen arama
from datetime import datetime


#pdf içerisinden görevleri çıkartan fonk
def extract_tasks_from_pdf(pdf_path):
    reader= PdfReader(pdf_path)
    text= "\n".join(page.extract_text() for page in reader.pages)
    #print(text)

    # Sadece görev tablosunun olduğu kısmı al
    text = text.split("Zaman Ekip Üyesi Görev Tanımı", 1)[1]

    # Sayfa sonlarını temizle
    text = re.sub(r"FitMiniApp - Flash Demo Raporu Sayfa \d+ / \d+", "", text)

    # Tüm satır sonlarını boşluğa çevir
    text = re.sub(r"\s+", " ", text)

    pattern = r"(\d{2}:\d{2})\s+(\w+)\s+(.*?)(?=\d{2}:\d{2}|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    print(matches)

    #hepsini bir listede topla
    tasks=[]
    for match in matches:
        zaman, kisi, gorev= match #eşleşme yani match 3 parçaya ayrılıyor
        zaman= datetime.strptime(zaman, "%H:%M")
        tasks.append({
            "timestamp": zaman,
            "person": kisi.strip(),
            "task": gorev.strip()
        })
    return tasks


if __name__ == "__main__":
    path= "Rapor.pdf"
    try:
        tasks= extract_tasks_from_pdf(path)
        for task in tasks:
            print(f"{task["timestamp"]} -- {task["person"]}: {task["task"]}")
    except FileNotFoundError:
        print(f"{path} dosyası bulunamadı.")