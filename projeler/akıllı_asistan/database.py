import sqlite3
import os

#veritabanı dosyasının yolunu oluştur
DB_PATH= os.path.join("data", "assistant.db")

#veritabanı başlatan fonk.
def initialize_db():

    #data klasörü yoksa oluştur
    os.makedirs("data", exist_ok=True)

    #veritabanına bağlan ve dosya yoksa oluştur
    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()

    #tablolar yoksa oluştur
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calendar(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            event_date TEXT NOT NULL
            )
        """
    )

    #değişiklikleri kaydet
    conn.commit()

    #bağlantıyı kapat
    conn.close()


#veritabanına yeni not ekleme
def add_note(content):

    #veritabanına bağlan
    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()

    #content'i "notes" tablosuna ekle
    cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))

    conn.commit()
    conn.close()


#veritabanına yeni etkinlik ekleme
def add_event(event, event_date):

    #veritabanına bağlan
    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()

    #etkinlik ve tarihi "calendar" tablosuna ekle
    cursor.execute("INSERT INTO calendar (event, event_date) VALUES (?, ?)", (event, event_date))

    conn.commit()
    conn.close()


#tüm notları veritabanından sıralı bir şekilde getiren fonk(terminalde yazdırmak)
def get_notes():
    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()

    #notes tablosundan içerik ve tarih bilgilerini zaman sırasına göre getir
    cursor.execute("SELECT content, created_at FROM notes ORDER BY created_at DESC")

    #sonuçları liste olarak al
    notes= cursor.fetchall()

    conn.commit()
    conn.close()

    return notes


#tüm etkinlikleri veritabanından sıralı bir şekilde getiren fonk(terminalde yazdırmak)
def get_events():
    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()

    #calendaar tablosundan etkinlikleri tarihe göre sırala
    cursor.execute("SELECT event, event_date FROM calendar ORDER BY event_date")

    #sonuçları liste olarak al
    events= cursor.fetchall()

    conn.commit()
    conn.close()

    return events



if __name__ == "__main__":
    initialize_db()
    add_note("kitapları almayı unutma")
    add_event("OS final", "27.07.2026")

    print(f"Notes: {get_notes()}")
    print(f"Notes: {get_events()}")

