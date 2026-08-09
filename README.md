# 🧠 GYZTA Deep Learning & NLP

**Doğal Dil İşleme (NLP), Derin Öğrenme ve Üretken Yapay Zeka (Generative AI) alanlarında; klasik metin işleme tekniklerinden modern LLM tabanlı ajan mimarilerine (Agentic AI) uzanan uçtan uca bir öğrenme ve proje portföyü.**

Bu repo, NLP'nin temel yapı taşlarından (Bag of Words, TF-IDF) başlayıp, derin öğrenme tabanlı dizi modellerine (RNN, GRU, LSTM), Transformer/BERT tabanlı gelişmiş görevlere ve son olarak **RAG (Retrieval-Augmented Generation)**, **LangChain Agent** ve **Google Gemini / Ollama** entegrasyonlu 7 adet gerçek dünya uygulamasına kadar ilerleyen kademeli bir müfredat şeklinde organize edilmiştir.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-Agents%20%26%20RAG-1C3C3C">
  <img alt="Gemini" src="https://img.shields.io/badge/Google-Gemini%20API-4285F4?logo=google&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Not%20Specified-lightgrey">
</p>

---

## 📑 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Repo Mimarisi](#-repo-mimarisi)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [1. `metin_on_isleme/` — Metin Ön İşleme](#1-metin_on_isleme--metin-ön-i̇şleme)
- [2. `metin_isleme/` — Metin Temsili & Vektörleştirme](#2-metin_isleme--metin-temsili--vektörleştirme)
- [3. `nlp_temel_gorevleri/` — Temel NLP Görevleri](#3-nlp_temel_gorevleri--temel-nlp-görevleri)
- [4. `derin_ogrenme/` — Derin Öğrenme (RNN / GRU / LSTM)](#4-derin_ogrenme--derin-öğrenme-rnn--gru--lstm)
- [5. `gelismis_nlp_gorevleri/` — Transformer & BERT Tabanlı Görevler](#5-gelismis_nlp_gorevleri--transformer--bert-tabanlı-görevler)
- [6. `projeler/` — Uçtan Uca Uygulamalar](#6-projeler--uçtan-uca-uygulamalar)
  - [6.1 AI Agents — Çok Araçlı Yapay Zeka Asistanı](#61-ai_agents_yapay_zeka_asistanı--çok-araçlı-yapay-zeka-asistanı)
  - [6.2 Akıllı Asistan (Not & Etkinlik Yöneticisi)](#62-akıllı_asistan--not--etkinlik-yöneticisi)
  - [6.3 Akıllı Proje Yöneticisi](#63-akıllı_proje_yöneticisi--otonom-görev-takip-ajanı)
  - [6.4 Akıllı Turizm Rehberi](#64-akıllı_turizm_rehberi--yerel-llm-sohbet-botu)
  - [6.5 Doktor Asistanı](#65-doktor_asistanı--sağlık-danışma-chatbotu)
  - [6.6 Müşteri Destek Botu](#66-musteri_destek_botu--rag-tabanlı-sss-chatbotu)
  - [6.7 Sözleşme İnceleme Asistanı](#67-sozlesme_inceleme_asistanı--hukuki-doküman-rag)
- [Kurulum](#-kurulum)
- [Ortam Değişkenleri (.env)](#-ortam-değişkenleri-env)
- [Kullanılan Veri Setleri](#-kullanılan-veri-setleri)
- [Mimari Kavramlar & Öğrenme Yol Haritası](#-mimari-kavramlar--öğrenme-yol-haritası)

---

## 🎯 Proje Hakkında

`GYZTA_deep_learning`, doğal dil işleme ve derin öğrenmeyi **teoriden pratiğe** taşıyan, ilerleyen zorluk seviyesinde tasarlanmış bir çalışma deposudur. Her bir Python dosyası; amacını, izlenen adımları ve kullanılan teknolojileri açıklayan detaylı Türkçe docstring'ler ile başlar — bu sayede repo aynı zamanda kendi kendine yetebilen bir **öğrenme günlüğü / ders notu** işlevi görür.

Repo iki ana bölümden oluşur:

1. **Temel Kavramlar (Klasik NLP → Derin Öğrenme → Transformer):** `metin_on_isleme`, `metin_isleme`, `nlp_temel_gorevleri`, `derin_ogrenme`, `gelismis_nlp_gorevleri` klasörleri, NLP'nin temel taşlarını sırasıyla küçük, odaklı script'ler halinde öğretir.
2. **Uygulamalı Projeler (Generative AI / Agentic AI):** `projeler/` klasörü, Google Gemini, Ollama (Gemma3), LangChain, FAISS, RAG ve çoklu araç (tool) kullanan ajanlarla geliştirilmiş **7 adet bağımsız, çalıştırılabilir uygulama** içerir. Bu projeler; terminal tabanlı chatbot'lardan, FastAPI + Streamlit ile üretilmiş tam teşekküllü web servislerine kadar uzanır.

---

## 🗂 Repo Mimarisi

```
GYZTA_deep_learning/
│
├── metin_on_isleme/              # Ham metin temizleme, tokenizasyon, stemming/lemmatization
│   ├── 1_veri_temizleme.py
│   ├── 2_tokenizasyon.py
│   ├── 3_kok_ve_govde_bulma.py
│   └── 4_durdurma_kelimeler.py
│
├── metin_isleme/                 # Metin temsili: BoW, TF-IDF, N-gram, Word Embedding
│   ├── 1_bow.py
│   ├── 2_bow_imdb.py
│   ├── 3_tf_idf.py
│   ├── 4_tf_idf_sms_spam.py
│   ├── 5_n_grams.py
│   ├── 6_word_embeddings.py
│   ├── 7_word_embedding_imdb.py
│   ├── IMDB Dataset.csv
│   └── sms_spam.csv
│
├── nlp_temel_gorevleri/          # Klasik ML + NLP: sınıflandırma, NER, POS, öneri sistemi
│   ├── duygu_analizi.py
│   ├── kelime_anlami_belirsizligi_giderme.py
│   ├── metin_parcasi_etiketleme.py
│   ├── metin_siniflandirma.py
│   ├── morfolojik_analiz.py
│   ├── oneri_sistemi.py
│   ├── varlik_ismi_tanima.py
│   └── amazon.csv / sms_spam.csv
│
├── derin_ogrenme/                # Keras/TensorFlow tabanlı dizi (sequence) modelleri
│   ├── 1_rnn.py                  # SimpleRNN ile duygu analizi
│   ├── 2_grn.py                  # GRU ile IMDB duygu analizi
│   └── 3_lstm.py                 # LSTM ile metin üretimi (text generation)
│
├── gelismis_nlp_gorevleri/       # HuggingFace Transformers / BERT tabanlı görevler
│   ├── 1_metin_ozetleme.py       # Summarization pipeline
│   ├── 2_soru_cevap_sistemi.py   # BERT + SQuAD Question Answering
│   ├── 3_bilgi_getirme.py        # BERT embedding ile semantic similarity
│   ├── 4_metin_cevirisi.py       # MarianMT ile EN→FR çeviri
│   └── requirement.txt
│
├── projeler/                     # Üretken yapay zeka (LLM) tabanlı 7 uçtan uca proje
│   ├── ai_agents_yapay_zeka_asistanı/
│   ├── akıllı_asistan/
│   ├── akıllı_proje_yöneticisi/
│   ├── akıllı_turizm_rehberi/
│   ├── doktor_asistanı/
│   ├── musteri_destek_botu/
│   └── sozlesme_inceleme_asistanı/
│
├── .vscode/                      # Editör ayarları
└── .gitignore
```

> **Not:** `metin_isleme/venv` altında commit edilmiş bir sanal ortam (virtualenv) bulunmaktadır. `.gitignore` bu klasörü hariç tutacak şekilde yapılandırılmış olsa da geçmişte repoya dahil edilmiş görünüyor; bu klasörü kullanmak yerine kendi sanal ortamınızı oluşturmanız önerilir (bkz. [Kurulum](#-kurulum)).

---

## 🛠 Kullanılan Teknolojiler

| Katman | Teknolojiler |
|---|---|
| **Dil** | Python 3.11+ |
| **Klasik ML / Veri İşleme** | scikit-learn, pandas, numpy |
| **NLP Araç Kutusu** | NLTK (tokenizasyon, stemming, lemmatization, VADER, WSD/Lesk, stopwords), spaCy (`en_core_web_sm` — POS tagging, NER, morfolojik analiz) |
| **Kelime Vektörleri** | Gensim (Word2Vec, FastText), scikit-learn (`CountVectorizer`, `TfidfVectorizer`) |
| **Derin Öğrenme** | TensorFlow / Keras (`SimpleRNN`, `GRU`, `LSTM`, `Embedding`, `Dense`, IMDB dataset) |
| **Transformer Modelleri** | HuggingFace `transformers` (BERT, `bert-large-uncased-whole-word-masking-finetuned-squad`, MarianMT / Helsinki-NLP `opus-mt-en-fr`), PyTorch backend |
| **Büyük Dil Modelleri (LLM)** | Google Gemini API (`gemini-2.5-flash`, `gemini-3.1-flash`, `gemini-3.5-flash-lite`, `gemini-1.5-flash`), Ollama üzerinden yerel çalıştırılan **Gemma 3 (4B)** |
| **Agent / Orkestrasyon** | LangChain, `langchain-classic`, `langchain-community`, `langchain-google-genai`, `langchain-ollama`, LangGraph, `ConversationBufferMemory`, `ZERO_SHOT_REACT_DESCRIPTION` agent tipi |
| **RAG & Vektör Veritabanları** | FAISS (`faiss-cpu`), `ConversationalRetrievalChain`, `RecursiveCharacterTextSplitter` |
| **Embedding Modelleri** | `sentence-transformers/LaBSE` (çok dilli), `all-MiniLM-L6-v2`, HuggingFace `sentence-transformers` |
| **PDF İşleme** | `PyPDFLoader` (LangChain), `PyPDF2`, `PyMuPDF` (`fitz`) |
| **Web Arama Aracı** | SerpAPI (`SerpAPIWrapper`) |
| **Backend / API** | FastAPI, Uvicorn, Pydantic |
| **Frontend / Arayüz** | Streamlit (chat arayüzü, dosya yükleme, session state, streaming callback) |
| **Veritabanı** | SQLite3 (`sqlite3`) |
| **Görselleştirme** | Matplotlib, PCA / KMeans (scikit-learn) — kelime vektörü görselleştirme ve kümeleme |
| **Yardımcı Kütüphaneler** | `python-dotenv`, `requests`, `rich` (renkli terminal çıktısı), `re` (regex) |

---

## 1. `metin_on_isleme/` — Metin Ön İşleme

NLP pipeline'ının ilk adımı olan ham metin temizleme tekniklerini kapsar.

| Dosya | Açıklama |
|---|---|
| `1_veri_temizleme.py` | Fazla boşlukları temizleme, büyük/küçük harf normalizasyonu, noktalama ve özel karakter temizliği, HTML etiketi kaldırma, yazım hatası düzeltme (`textblob`, `beautifulsoup4`) |
| `2_tokenizasyon.py` | NLTK `punkt` ile kelime (word) ve cümle (sentence) tokenizasyonu |
| `3_kok_ve_govde_bulma.py` | **Stemming** (`PorterStemmer`) ve **Lemmatization** (`WordNetLemmatizer`) karşılaştırması |
| `4_durdurma_kelimeler.py` | İngilizce, Türkçe ve manuel stop-word (durdurma kelimesi) çıkarma yöntemleri |

---

## 2. `metin_isleme/` — Metin Temsili & Vektörleştirme

Metni makine öğrenmesi modellerinin işleyebileceği sayısal forma dönüştüren klasik ve modern yöntemler.

| Dosya | Açıklama | Veri Seti |
|---|---|---|
| `1_bow.py` | `CountVectorizer` ile temel **Bag of Words** temsili | Örnek metinler |
| `2_bow_imdb.py` | CSV okuma → metin temizleme (regex, stopwords) → BoW → en sık geçen 5 kelime | `IMDB Dataset.csv` |
| `3_tf_idf.py` | `TfidfVectorizer` ile **TF-IDF** ağırlıklandırma, ortalama tf-idf skorları | Örnek belgeler |
| `4_tf_idf_sms_spam.py` | SMS verisi üzerinde TF-IDF, en yüksek skorlu 10 kelimenin çıkarılması | `sms_spam.csv` |
| `5_n_grams.py` | Unigram, bigram, trigram karşılaştırmalı analiz | Örnek cümleler |
| `6_word_embeddings.py` | **Word2Vec** (Google) ve **FastText** (Meta) ile embedding eğitimi, PCA ile 3B görselleştirme | Örnek cümleler |
| `7_word_embedding_imdb.py` | IMDB yorumlarında Word2Vec embedding → **KMeans** ile pozitif/negatif kümeleme → PCA ile 2B görselleştirme | `IMDB Dataset.csv` |

---

## 3. `nlp_temel_gorevleri/` — Temel NLP Görevleri

Klasik makine öğrenmesi ve kural/sözlük tabanlı NLP görevlerinin uygulamaları.

| Dosya | Açıklama | Teknik |
|---|---|---|
| `duygu_analizi.py` | Amazon yorumlarının pozitif/negatif sınıflandırılması | NLTK **VADER** (`SentimentIntensityAnalyzer`), confusion matrix & classification report |
| `kelime_anlami_belirsizligi_giderme.py` | Aynı kelimenin (örn. *bank*) bağlama göre farklı anlamlarının çözümlenmesi | **Lesk algoritması** (`nltk.wsd.lesk`), WordNet |
| `metin_parcasi_etiketleme.py` | Cümledeki her kelimenin dilbilgisel türünün (POS) etiketlenmesi | spaCy `en_core_web_sm` |
| `metin_siniflandirma.py` | SMS Spam Collection üzerinde spam/ham (normal) ikili sınıflandırma | BoW + **Decision Tree Classifier** |
| `morfolojik_analiz.py` | Kelimenin kökü (lemma), POS kategorisi, morfolojik özellikleri (tekil/çoğul vb.) | spaCy morfoloji modülü |
| `oneri_sistemi.py` | Kullanıcı-ürün derecelendirmelerinden öğrenen bir öneri motoru | Embedding tabanlı **Neural Collaborative Filtering** (Keras `Dot`, `Embedding`) |
| `varlik_ismi_tanima.py` | Metindeki kişi, organizasyon, yer gibi özel varlıkların tespiti | spaCy **NER** (`doc.ents`) |

---

## 4. `derin_ogrenme/` — Derin Öğrenme (RNN / GRU / LSTM)

TensorFlow/Keras ile dizi (sequence) verisi üzerinde çalışan derin öğrenme mimarileri.

| Dosya | Görev | Mimari | Detaylar |
|---|---|---|---|
| `1_rnn.py` | Restoran yorumlarında duygu analizi (ikili sınıflandırma) | `Embedding` (Word2Vec önceden eğitilmiş ağırlıklarla, `trainable=False`) → `SimpleRNN` → `Dense(sigmoid)` | Kendi oluşturulan 50 cümlelik simüle veri seti; `Tokenizer`, `pad_sequences`, `LabelEncoder`, `train_test_split` |
| `2_grn.py` | IMDB film yorumu duygu analizi | `Embedding` → `GRU(64)` → `Dense(sigmoid)` | Keras yerleşik `imdb` veri seti (en sık 10.000 kelime), `padding='pre'`, `word_index` ↔ `index_to_word` çevrimi ile insan-okunabilir çözümleme |
| `3_lstm.py` | Türkçe metin üretimi (text generation / dil modeli) | `Embedding` → `LSTM(100)` → `Dense(softmax)` | N-gram dizi üretimi, one-hot encoding (`to_categorical`), `generate_text()` fonksiyonu ile seed metinden kelime üretimi |

---

## 5. `gelismis_nlp_gorevleri/` — Transformer & BERT Tabanlı Görevler

HuggingFace `transformers` kütüphanesi ile önceden eğitilmiş (pretrained) büyük modellerin pipeline ve doğrudan model/tokenizer API'leri üzerinden kullanımı.

| Dosya | Görev | Model |
|---|---|---|
| `1_metin_ozetleme.py` | Uzun metinlerin otomatik özetlenmesi | HuggingFace `pipeline("summarization")` |
| `2_soru_cevap_sistemi.py` | Verilen bir bağlam (context) içinden soruya cevap bulma (extractive QA) | `bert-large-uncased-whole-word-masking-finetuned-squad` (SQuAD ile fine-tune edilmiş BERT) |
| `3_bilgi_getirme.py` | Sorgu ile belgeler arasında **anlamsal benzerlik** (semantic similarity) ölçümü | `bert-base-uncased` embedding + kosinüs benzerliği (`cosine_similarity`) |
| `4_metin_cevirisi.py` | İngilizce → Fransızca nöral makine çevirisi (NMT) | `Helsinki-NLP/opus-mt-en-fr` (MarianMT, seq2seq mimarisi) |

Bağımlılıklar `requirement.txt` içinde tanımlıdır (`transformers==4.39.3`, `torch==2.13.0`, `scikit-learn`, `sentencepiece` vb.).

---

## 6. `projeler/` — Uçtan Uca Uygulamalar

Bu klasör, repodaki en kapsamlı bölümdür: her biri kendi `requirements.txt` dosyasına sahip, **bağımsız çalıştırılabilir 7 gerçek dünya uygulaması** içerir. Ortak tema; Google Gemini veya yerel Ollama/Gemma3 modellerinin, **RAG**, **hafıza (memory)** ve **çoklu araç (tool) kullanımı** ile birleştirilerek pratik problemlere uygulanmasıdır.

### 6.1 `ai_agents_yapay_zeka_asistanı/` — Çok Araçlı Yapay Zeka Asistanı

Repodaki en gelişmiş proje. **LangChain ReAct Agent** mimarisi kullanarak kullanıcı isteğine göre otonom şekilde doğru aracı seçen, çok yetenekli bir yapay zeka ajanı.

- **LLM:** Google Gemini (`gemini-3.5-flash-lite`)
- **Agent tipi:** `ZERO_SHOT_REACT_DESCRIPTION` (`langchain_classic.agents.initialize_agent`)
- **Araçlar (Tools):**
  | Araç | Dosya | İşlev |
  |---|---|---|
  | `RAGTool` | `tools/rag_tool.py` | `sss.pdf` belgesini FAISS'e vektörleştirip `ConversationalRetrievalChain` ile belge tabanlı soru-cevap yapar (embedding: `sentence-transformers/LaBSE`) |
  | `calculator` | `tools/calculator_tool.py` | `@tool` dekoratörü ile matematiksel ifadeleri değerlendirir (`eval`) |
  | `discount_calculator` | `tools/custom_discount_tool.py` | Metinden regex ile fiyat çıkarıp %10 indirim uygular |
  | `SearchTool` | `main_agent.py` | SerpAPI (`SerpAPIWrapper`) üzerinden web/Google araması yapar |
  | Memory | `ConversationBufferMemory` | Kullanıcı ile önceki konuşmaları hatırlar |
- **Sunum katmanları:**
  - `main_agent.py` — Terminal tabanlı çalıştırma
  - `fast_api.py` — `/ask` endpoint'i ile REST API (FastAPI + Uvicorn)
  - `client.py` — API'yi test etmek için basit `requests` istemcisi
  - `app_streamlit.py` — FastAPI backend'e bağlanan Streamlit sohbet arayüzü
- **Veri:** `data/sss.pdf` (RAG kaynağı)
- **Sistem akışı:** `Streamlit → FastAPI /ask → LangChain Agent (tool seçimi) → Gemini (reasoning) → Memory güncelleme → yanıt`

### 6.2 `akıllı_asistan/` — Not & Etkinlik Yöneticisi

Doğal dilde konuşulabilen, notlarını ve takvim etkinliklerini SQLite'ta saklayan kişisel asistan.

- **LLM:** Google Gemini (`gemini-3.1-flash-lite`, doğrudan REST API çağrısı — `requests` ile)
- **Modüller:**
  - `database.py` — SQLite3 ile `notes` ve `calendar` tablolarının CRUD işlemleri (`initialize_db`, `add_note`, `add_event`, `get_notes`, `get_events`)
  - `asisstant.py` — Gemini API'ye HTTP POST isteği gönderen `get_gemini_response()` ve kullanıcı niyetini (intent) sınıflandıran `detect_intent()`
  - `main.py` — Komut satırı arayüzü: `not ekle | etkinlik ekle | notları göster | etkinlikleri göster | sohbet et | çıkış`
- **Öne çıkan özellik:** Kullanıcı "sohbet et" komutunu verdiğinde, mesajın niyeti (`not_ozet` / `etkinlik_ozet` / genel sohbet) tespit edilir ve ilgili veriler prompt'a bağlam olarak eklenerek Gemini'ye gönderilir.
- **Veri:** `data/assistant.db` (SQLite veritabanı)

### 6.3 `akıllı_proje_yöneticisi/` — Otonom Görev Takip Ajanı

Bir proje planı PDF'ini okuyup, ekip üyelerine **zamanlanmış görev hatırlatmaları** yapan ve doğal dildeki yanıtları analiz eden simülasyon tabanlı bir yapay zeka proje yöneticisi.

- **LLM:** Google Gemini (`gemini-1.5-flash`, `google-generativeai` SDK)
- **Modüller:**
  - `pdf_reader.py` — `PyPDF2` ile PDF'den metin çıkarma; regex (`re.findall`) ile "saat — kişi — görev" üçlülerinin ayrıştırılması (`extract_tasks_from_pdf`)
  - `gemini_agent.py` — Kişi, görev, zaman ve geçmiş yanıtlara göre kişiselleştirilmiş takip sorusu üreten `generate_followup_questions()`, görevin tamamlanıp tamamlanmadığını doğal dilden anlayan `is_task_completed()`
  - `scheduler_gemini_manager.py` — 10 saniyede 1 dakika ilerleyen bir **zaman simülasyonu** ile görevlerin sırayla sorulmasını yöneten ana döngü (`run_scheduler`)
- **Arayüz:** `rich` kütüphanesi ile renkli/biçimli terminal çıktısı
- **Veri:** `Rapor.pdf` (proje planı/takvimi)

### 6.4 `akıllı_turizm_rehberi/` — Yerel LLM Sohbet Botu

Türkiye turizmi hakkında soru-cevap yapan, **tamamen yerelde (on-premise)** çalışan bir chatbot — veriler buluta gönderilmez.

- **LLM:** **Ollama** üzerinden çalıştırılan **Gemma 3 (4B parametre)** — `ChatOllama(model="gemma3:4b")`
- **Framework:** LangChain (`SystemMessage`, `HumanMessage`, `ConversationBufferMemory`)
- **Dosyalar:**
  - `akilli_turizm_rehberi_terminal.py` — Terminal tabanlı sohbet döngüsü
  - `akıllı_turizm_rehberi_streamlit.py` — Streamlit web arayüzü (session state ile hafıza)
  - `streamlit_streaming.py` — Özel `StreamHandler` (`BaseCallbackHandler`) sınıfı ile **token-token canlı yanıt akışı (streaming)** eklenmiş gelişmiş versiyon
- **Neden Ollama?** Kullanım senaryosu, verilerin buluta gitmemesi gereken ve eşzamanlı kullanıcı sayısının düşük olduğu durumlar için tasarlanmıştır; yerel modeller (Llama, Mistral, Gemma, Qwen, DeepSeek) bu şekilde çalıştırılabilir.

### 6.5 `doktor_asistanı/` — Sağlık Danışma Chatbotu

Kullanıcının adı, yaşı ve sorusuna göre kişiselleştirilmiş sağlık tavsiyeleri üreten, **RAG kullanmayan** (saf prompt engineering tabanlı), çok kullanıcılı bellek yönetimine sahip bir chatbot.

- **LLM:** Google Gemini (`gemini-2.5-flash`, `langchain_google_genai.ChatGoogleGenerativeAI`)
- **Dosyalar:**
  - `doktor_asistani_terminal.py` — Terminal üzerinden ilk prototip
  - `doktor_asistani_api.py` — FastAPI ile `/chat` endpoint'i; her kullanıcı (`name`) için ayrı `ConversationBufferMemory` nesnesi tutan `user_memories: Dict[str, ConversationBufferMemory]` yapısı; `ConversationChain` ile hafıza+model zinciri
  - `client_test.py` — `requests` ile terminal tabanlı test istemcisi (isim/yaş alır, sohbeti sürdürür)
- **Test yöntemi:** Uygulama hem `client_test.py` ile hem de Swagger UI (`/docs`) üzerinden test edilebilir.

### 6.6 `musteri_destek_botu/` — RAG Tabanlı SSS Chatbotu

Sık Sorulan Sorular (SSS) PDF'ine dayalı, Türkçe destekli bir müşteri destek asistanı.

- **LLM:** Ollama üzerinden **Gemma 3 (4B)** (`temperature=0.2` — daha "garantici", düşük yaratıcılık)
- **RAG Pipeline:**
  - `load_pdf_and_embedding.py` — `sss.pdf` → `PyPDFLoader` → `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50) → `HuggingFaceEmbeddings` (`sentence-transformers/LaBSE`, çok dilli) → **FAISS** vektör veritabanı → yerel diske kayıt (`sss_store/`)
  - `chatbot_rag_memory.py` — Kayıtlı FAISS index'i yükler, `ConversationalRetrievalChain` (LLM + Memory + VectorDB) ile terminal tabanlı soru-cevap sağlar
  - `streamlit_app.py` — Kullanıcının **kendi PDF'ini yükleyebildiği** dinamik bir Streamlit arayüzü; geçici dosya (`tempfile`) üzerinden anlık chunking + embedding + FAISS index oluşturma
- **Önceden hazırlanmış vektör veritabanı:** `sss_store/index.faiss`, `sss_store/index.pkl`

### 6.7 `sozlesme_inceleme_asistanı/` — Hukuki Doküman RAG

Yüklenen bir sözleşme PDF'inden bilgi çıkaran, "sözleşme avukatı" rolündeki bir RAG asistanı.

- **LLM:** Google Gemini (`gemini-3.0-flash`, `google.generativeai.GenerativeModel`)
- **RAG Pipeline:**
  - `build_vector_db.py` — `sozlesme_ornek.pdf` → **PyMuPDF (`fitz`)** ile metin çıkarma → satır bazlı özel `chunk_text()` fonksiyonu (max 500 karakter) → `SentenceTransformer("all-MiniLM-L6-v2")` ile embedding → **FAISS `IndexFlatL2`** (Euclidean/L2 mesafesi) index → `pickle` ile chunk'ların ve `.faiss` dosyasının diske kaydı
  - `main.py` — Soru embedding'e çevrilir → FAISS'ten en yakın `k=3` chunk getirilir → İngilizce hukuki prompt şablonu ile bağlam + soru Gemini'ye gönderilir → yanıt üretilir
- **Örnek veri:** `data/sozlesme_ornek.pdf`, önceden oluşturulmuş `data/sozlesme_ornek.faiss` ve `data/sozlesme_ornek.pkl`
- **Örnek Q&A senaryosu** dosya içinde belgelenmiştir (API maliyeti sorumluluğu, hatalı LLM çıktısı sorumluluğu, bakım kapsamı gibi sözleşme maddeleri üzerinden).

---

## ⚙️ Kurulum

Her proje/klasör kendi `requirements.txt` dosyasına sahip olduğundan, **her modül için ayrı bir sanal ortam kullanılması önerilir** (bağımlılık çakışmalarını önlemek için — örneğin `langchain` sürümleri klasörler arasında farklılık göstermektedir: bazı projeler `langchain 0.3.x`, bazıları `langchain 1.3.x / langchain-classic` kullanır).

```bash
# Depoyu klonlayın
git clone https://github.com/Ahsen-Nur/GYZTA_deep_learning.git
cd GYZTA_deep_learning

# İlgilendiğiniz klasöre girin, örn:
cd projeler/ai_agents_yapay_zeka_asistanı

# Sanal ortam oluşturun
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### spaCy modeli (nlp_temel_gorevleri için)

```bash
python -m spacy download en_core_web_sm
```

### NLTK veri paketleri (metin_on_isleme / nlp_temel_gorevleri için)

İlgili script'ler `nltk.download(...)` çağrılarını kod içinde otomatik yapar (`punkt`, `wordnet`, `omw-1.4`, `stopwords`, `vader_lexicon`).

### Ollama tabanlı projeler için (akıllı_turizm_rehberi, musteri_destek_botu)

```bash
# Ollama'yı kurun: https://ollama.com
ollama pull gemma3:4b
ollama serve
```

### Örnek çalıştırma komutları

```bash
# Terminal tabanlı ajan
python main_agent.py

# FastAPI servisini başlatma
uvicorn fast_api:app --reload

# Streamlit arayüzü
streamlit run app_streamlit.py
```

---

## 🔑 Ortam Değişkenleri (.env)

LLM tabanlı projeler `.env` dosyası üzerinden API anahtarı okur (`python-dotenv`). Her proje klasöründe kendi `.env` dosyanızı oluşturmanız gerekir (repoya dahil edilmemiştir):

```env
# Google Gemini API anahtarı (çoğu proje için gerekli)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# SerpAPI anahtarı (sadece ai_agents_yapay_zeka_asistanı — web arama aracı için)
SERP_API_KEY=your_serpapi_key_here
```

> Gemini API anahtarı [Google AI Studio](https://aistudio.google.com/) üzerinden, SerpAPI anahtarı ise [serpapi.com](https://serpapi.com/) üzerinden alınabilir.

---

## 📊 Kullanılan Veri Setleri

| Veri Seti | Konum | Kullanım Amacı |
|---|---|---|
| `IMDB Dataset.csv` | `metin_isleme/` | Film yorumu duygu analizi (BoW, Word2Vec + KMeans) |
| `sms_spam.csv` | `metin_isleme/`, `nlp_temel_gorevleri/` | Spam/ham SMS sınıflandırma (TF-IDF, Decision Tree) |
| `amazon.csv` | `nlp_temel_gorevleri/` | Amazon ürün yorumu duygu analizi (VADER) |
| Keras `imdb` (yerleşik) | `derin_ogrenme/2_grn.py` | GRU tabanlı duygu analizi (en sık 10.000 kelime) |
| Simüle edilmiş restoran yorumları | `derin_ogrenme/1_rnn.py` | RNN eğitimi için LLM (GPT/Gemini) yardımıyla üretilmiş 50 örnek |
| Simüle edilmiş Türkçe günlük cümleler | `derin_ogrenme/3_lstm.py` | LSTM tabanlı metin üretimi eğitimi |
| `sss.pdf` | `ai_agents_yapay_zeka_asistanı/`, `musteri_destek_botu/` | RAG kaynak belgesi (SSS) |
| `Rapor.pdf` | `akıllı_proje_yöneticisi/` | Simülasyon için proje/görev takvimi |
| `sozlesme_ornek.pdf` | `sozlesme_inceleme_asistanı/` | Örnek hukuki sözleşme (RAG kaynağı) |

---

## 🧭 Mimari Kavramlar & Öğrenme Yol Haritası

Repo, aşağıdaki kavramsal ilerlemeyi takip edecek şekilde tasarlanmıştır:

```
Ham Metin
   │
   ▼
[metin_on_isleme]  → Temizlik, Tokenizasyon, Stemming/Lemmatization, Stopwords
   │
   ▼
[metin_isleme]     → Sayısal temsil: BoW → TF-IDF → N-gram → Word2Vec/FastText
   │
   ▼
[nlp_temel_gorevleri] → Klasik ML ile NLP: Sınıflandırma, NER, POS, WSD, Öneri Sistemleri
   │
   ▼
[derin_ogrenme]    → Sıralı veri modelleme: RNN → GRU → LSTM (sınıflandırma & üretim)
   │
   ▼
[gelismis_nlp_gorevleri] → Transformer / BERT: Özetleme, QA, Semantic Search, Çeviri
   │
   ▼
[projeler]         → Generative AI: LLM + RAG + Agent + Memory + Tool-Use
                       (Gemini / Gemma3 · LangChain · FAISS · FastAPI · Streamlit)
```

Bu ilerleme; **istatistiksel temsillerden (BoW/TF-IDF) → öğrenilmiş temsillere (embedding) → sıralı sinir ağlarına (RNN/LSTM) → dikkat mekanizmalı Transformer modellerine → son olarak bu modelleri gerçek zamanlı, hafızalı ve araç kullanan ajanlar haline getiren LLM orkestrasyonuna** kadar olan modern NLP tarihini pratik olarak yeniden üretir.

**Projeler bölümünde tekrar eden mimari desenler:**
- **RAG (Retrieval-Augmented Generation):** PDF → Chunking → Embedding → Vektör DB (FAISS) → Retrieval → LLM ile yanıt üretimi (`ai_agents`, `musteri_destek_botu`, `sozlesme_inceleme_asistanı`)
- **Conversational Memory:** `ConversationBufferMemory` ile çok turlu, bağlamı koruyan sohbetler (`ai_agents`, `akıllı_turizm_rehberi`, `doktor_asistanı`, `musteri_destek_botu`)
- **Multi-Tool Agent (ReAct):** LLM'in kullanıcı isteğine göre uygun aracı (RAG, hesap makinesi, web arama vb.) otonom seçmesi (`ai_agents_yapay_zeka_asistanı`)
- **API + UI Ayrımı:** FastAPI (backend/mantık) + Streamlit (kullanıcı arayüzü) üzerinden servis mimarisi (`ai_agents`, `doktor_asistanı`, `musteri_destek_botu`)
- **Yerel (On-Prem) vs Bulut LLM:** Ollama/Gemma3 (veri gizliliği, düşük maliyet) ile Gemini API (ölçeklenebilirlik, güç) arasındaki mimari tercih karşılaştırması

---

