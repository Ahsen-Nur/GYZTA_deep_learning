from PyPDF2 import PdfReader
import re
from datetime import datetime


def extract_tasks_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages)

    text = text.split("Zaman Ekip Üyesi Görev Tanımı", 1)[1]
    text = re.sub(r"FitMiniApp - Flash Demo Raporu Sayfa \d+ / \d+", "", text)
    text = re.sub(r"Not: Bu belge.*?zamana yayılmalıdır\.", "", text, flags=re.DOTALL)
    text = text.replace("Zaman Ekip Üyesi Görev Tanımı", "")  # 2. sayfa başlığı tekrarı

    kisiler = "Kaan|Can|Yılmaz|Tüm Ekip"
    pattern = rf"(\d{{2}}:\d{{2}})\s*({kisiler})\s*(.*?)(?=\n\d{{2}}:\d{{2}}|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    tasks = []
    for match in matches:
        zaman, kisi, gorev = match
        zaman = datetime.strptime(zaman, "%H:%M")
        gorev = re.sub(r"\s+", " ", gorev).strip()  # satır içi \n'leri boşluğa çevir
        tasks.append({
            "timestamp": zaman,
            "person": kisi.strip(),
            "task": gorev
        })
    return tasks


if __name__ == "__main__":
    path = "Rapor.pdf"
    try:
        tasks = extract_tasks_from_pdf(path)
        for task in tasks:
            print(f'{task["timestamp"]} -- {task["person"]}: {task["task"]}')
    except FileNotFoundError:
        print(f"{path} dosyası bulunamadı.")