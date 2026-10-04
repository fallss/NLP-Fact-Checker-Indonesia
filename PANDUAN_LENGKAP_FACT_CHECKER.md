# 📖 PANDUAN LENGKAP SISTEM — INDONESIAN NEWS FACT-CHECKER v2.0
### Panduan Praktis Arsitektur, Cara Kerja, dan Penggunaan untuk Pengembang & Pengguna

---

> [!NOTE]
> Panduan ini disusun untuk membantu siapa pun (dosen, penguji, rekan tim, maupun pengguna baru) memahami cara kerja sistem **Indonesian News Fact-Checker** dari tingkat konsep hingga eksekusi teknis langkah-demi-langkah.

---

## 📑 DAFTAR ISI
1. [Sekilas tentang Proyek (Konsep dalam 2 Menit)](#1-sekilas-tentang-proyek-konsep-dalam-2-menit)
2. [Alur Kerja Sistem dari Hulu ke Hilir (End-to-End Walkthrough)](#2-alur-kerja-sistem-dari-hulu-ke-hilir-end-to-end-walkthrough)
3. [Panduan Instalasi & Persiapan Lingkungan](#3-panduan-instalasi--persiapan-lingkungan)
4. [Cara Menjalankan Sistem (3 Mode Pilihan)](#4-cara-menjalankan-sistem-3-mode-pilihan)
   - [Mode 1: Modern Web App (Gradio Dashboard)](#mode-1-modern-web-app-gradio-dashboard)
   - [Mode 2: Jupyter Notebook Interaktif](#mode-2-jupyter-notebook-interaktif)
   - [Mode 3: Terminal Command-Line Interface (CLI)](#mode-3-terminal-command-line-interface-cli)
   - [Mode 4: Penggunaan sebagai Library Python](#mode-4-penggunaan-sebagai-library-python)
5. [Bedah Antarmuka Web App & Makna Setiap Elemen UI](#5-bedah-antarmuka-web-app--makna-setiap-elemen-ui)
6. [10 Skenario Pengujian Nyata & Hasilnya](#6-10-skenario-pengujian-nyata--hasilnya)
7. [Struktur File & Penjelasan Setiap Skrip](#7-struktur-file--penjelasan-setiap-skrip)
8. [Pertanyaan Umum (FAQ) & Pemecahan Masalah](#8-pertanyaan-umum-faq--pemecahan-masalah)

---

## 1. Sekilas tentang Proyek (Konsep dalam 2 Menit)

### Apa Masalah yang Diselesaikan?
Di era informasi digital, hoaks dan disinformasi menyebar dalam hitungan menit di media sosial. Memverifikasi fakta secara manual membutuhkan waktu lama dan tenaga manusia yang besar.

### Bagaimana Solusinya?
Sistem ini bertindak layaknya seorang **jurnalis pemeriksa fakta digital (*automated fact-checker*)**:
1. **Menerima Klaim**: Pengguna memasukkan sebuah klaim berita bahasa Indonesia (misal: *"Presiden Jokowi melarang Wapres Ma'ruf Amin mengunjungi lokasi kebakaran Plumpang"*).
2. **Mencari Bukti (*Retrieval*)**: Sistem secara kilat menyisir **32.000 artikel berita** dari 7 media nasional tepercaya (Detik, Kompas, Tempo, CNN Indonesia, Republika, Antara, Kumparan) untuk mencari artikel dan potongan kalimat (*passages*) yang paling relevan.
3. **Menganalisis Hubungan Logika (*NLI*)**: Menggunakan model kecerdasan buatan canggih **mDeBERTa-v3 Multilingual**, sistem membandingkan klaim dengan kalimat bukti:
   - Apakah bukti **mendukung** klaim? (*Entailment*)
   - Apakah bukti **membantah** klaim? (*Contradiction*)
   - Atau berita **tidak cukup membahas** klaim tersebut? (*Neutral*)
4. **Memberikan Putusan (*Verdict*)**:
   - `✔ DIDUKUNG FAKTA` (*SUPPORTED*)
   - `✘ BERTENTANGAN / HOAKS` (*REFUTED*)
   - `? BUKTI TIDAK CUKUP` (*NOT ENOUGH INFO*)
5. **Menjelaskan Alasannya (*Explainability / XAI*)**: Menggunakan algoritma **SHAP**, sistem menyoroti kata-kata spesifik pada bukti berita yang menyebabkan AI mengambil kesimpulan tersebut.

---

## 2. Alur Kerja Sistem dari Hulu ke Hilir (End-to-End Walkthrough)

Mari kita bedah apa yang terjadi di balik layar saat Anda menekan tombol **"⚡ Periksa Fakta"**:

```text
 [ Klaim Pengguna ]
        │
        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1. PEMBERSIHAN KLAIM                                                        │
 │    Normalisasi spasi, pembersihan karakter non-standar.                     │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 2. HYBRID RETRIEVAL (Pencarian Artikel)                                     │
 │    a. BM25 Sparse Search: Mencari kemiripan kata kunci persis, nama tokoh,  │
 │       dan angka (misal: "Jokowi", "Ma'ruf Amin", "Plumpang").               │
 │    b. Sentence-BERT Dense Search: Mengubah klaim jadi vektor semantik 384-D │
 │       dan mencari artikel yang mirip secara makna (parafrase).              │
 │    c. Reciprocal Rank Fusion (RRF): Menggabungkan kedua hasil pemeringkatan │
 │       menjadi Top-8 artikel teratas.                                        │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 3. PASSAGE SELECTION (Pemilihan Bukti Kalimat)                              │
 │    Artikel utuh (300-500 kata) terlalu panjang untuk model NLI.             │
 │    Sistem memotong artikel menjadi jendela 3 kalimat (Sliding Window), lalu │
 │    memilih Top-5 passage yang paling relevan dengan klaim.                  │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 4. NATURAL LANGUAGE INFERENCE / NLI (Inferensi Logika)                      │
 │    Model mDeBERTa-v3 membaca [Premis: Bukti] dan [Hipotesis: Klaim].        │
 │    Menghasilkan 3 probabilitas untuk setiap passage:                        │
 │    P(Entailment), P(Neutral), P(Contradiction).                             │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 5. AGREGASI PUTUSAN (FEVER-style Max Aggregation)                           │
 │    - Memeriksa Gerbang Relevansi (Relevance Score >= 0.35).                 │
 │    - Mengambil skor kontradiksi dan entailment tertinggi (Max Pooling).     │
 │    - Jika P(Contradiction) >= 0.60 ──▶ BERTENTANGAN / HOAKS                 │
 │    - Jika P(Entailment) >= 0.60    ──▶ DIDUKUNG FAKTA                       │
 │    - Selain itu                    ──▶ BUKTI TIDAK CUKUP                    │
 │    - Menentukan "👑 Evidence Penentu" (bukti dengan skor paling dominan).   │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 6. EXPLAINABLE AI / SHAP (Transparansi Model)                               │
 │    Menganalisis kalimat pada Evidence Penentu. Menghitung kontribusi tiap   │
 │    kata terhadap pergeseran probabilitas verdict akhir.                     │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
                        ▼
 [ Tampilan Kartu Visual Interaktif (Web App / Notebook / Terminal) ]
```

---

## 3. Panduan Instalasi & Persiapan Lingkungan

### Prasyarat Sistem:
- **Sistem Operasi**: Windows 10/11, macOS, atau Linux.
- **Python**: Versi 3.10 atau lebih baru (disarankan 3.10–3.12).
- **RAM**: Minimal 8 GB (disarankan 16 GB).
- **GPU (Opsional)**: NVIDIA GPU dengan CUDA untuk eksekusi lebih cepat, namun sistem **100% dapat berjalan lancar di CPU**.

### Langkah 1: Buka Terminal & Aktifkan Virtual Environment
Buka PowerShell atau Command Prompt pada folder proyek ini:
```powershell
.\venv\Scripts\Activate.ps1
```
*(Tanda `(venv)` akan muncul di awal baris perintah terminal).*

### Langkah 2: Instalasi Dependensi
Pastikan seluruh pustaka pihak ketiga telah terpasang:
```powershell
pip install -r requirements.txt
```

### Langkah 3: Inisialisasi Indeks Pencarian (Sekali Saja)
Sistem membutuhkan indeks vektor pencarian dari 32.000 artikel berita. Bangun indeks sekali saja:
```powershell
python -m fact_checker build-index
```
> [!TIP]
> Proses pembuatan indeks membutuhkan waktu sekitar 8–10 menit di CPU (atau ~2 menit di GPU). Setelah selesai, indeks disimpan rapi dalam folder `cache/`. Pada eksekusi berikutnya, pemuatan indeks hanya memakan waktu **2 detik**!

---

## 4. Cara Menjalankan Sistem (3 Mode Pilihan)

### Mode 1: Modern Web App (Gradio Dashboard) — *Sangat Direkomendasikan*
Antarmuka grafis web interaktif yang modern, estetik, dan mudah digunakan:
```powershell
python -m fact_checker app
```
- Buka browser Anda di: **`http://127.0.0.1:7860`**
- *Fitur*: Masukkan klaim bebas atau klik contoh klaim yang tersedia, atur opsi (jumlah evidence, ambang batas), lihat verdict bercahaya, kartu bukti sumber asli dengan tautan (`↗`), dan highlight kata SHAP.
- *Opsi Berbagi Publik*: Tambahkan flag `--share` jika ingin membuat tautan publik sementara:
  ```powershell
  python -m fact_checker app --share
  ```

---

### Mode 2: Jupyter Notebook Interaktif
Bagi Anda yang ingin melihat alur eksperimen, eksplorasi data (EDA), pembersihan teks, grafik evaluasi, hingga menjalankan Web App di dalam sel:
1. Buka file [notebook_version/Fact_Checker_Full_Project.ipynb](file:///c:/Users/IFHAL%20FAIZI/Downloads/KULIAH/SEMESTER%207/NLP/Project%20UTS/notebook_version/Fact_Checker_Full_Project.ipynb) di VS Code atau Jupyter Lab.
2. Pilih kernel `Python 3 (venv)`.
3. Jalankan sel berurutan dari Bagian 1 hingga Bagian 13.
4. Di **Bagian 8**, hasil verifikasi ditampilkan dalam bentuk teks terminal dan **kartu visual HTML** modern.
5. Di **Bagian 12**, Anda dapat menjalankan antarmuka Gradio langsung di dalam notebook.

---

### Mode 3: Terminal Command-Line Interface (CLI)
Sangat cocok untuk pemeriksaan kilat langsung dari konsol tanpa membuka browser:

1. **Periksa Satu Klaim Tertentu**:
   ```powershell
   python -m fact_checker check "Presiden Jokowi memerintahkan Wapres Ma'ruf Amin meninjau lokasi kebakaran Plumpang."
   ```
2. **Mode Tanya-Jawab Interaktif (Looping Terminal)**:
   ```powershell
   python -m fact_checker interactive
   ```
3. **Mode Demo Otomatis (3 Kasus Uji Ekstrem)**:
   ```powershell
   python -m fact_checker demo
   ```
4. **Jalankan Evaluasi Benchmark**:
   ```powershell
   python -m fact_checker evaluate --compare-baseline
   ```

---

### Mode 4: Penggunaan sebagai Library Python
Anda dapat mengimpor package `fact_checker` ke dalam skrip Python buatan Anda sendiri:

```python
from fact_checker import FactChecker

# Inisialisasi pipeline pemeriksa fakta
fc = FactChecker()

# Periksa sebuah klaim
result = fc.check("FIFA mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20 2023.")

# Tampilkan ringkasan hasil
print(f"Verdict    : {result.verdict.label}")
print(f"Confidence : {result.verdict.confidence:.1%}")
print(f"Alasan     : {result.verdict.reason}")

# Tampilkan bukti-bukti yang ditemukan
for i, ev in enumerate(result.evidences, 1):
    penentu = " (👑 PENENTU)" if ev.is_decisive else ""
    print(f"\nBukti #{i}{penentu}:")
    print(f"  Sumber   : {ev.source} ({ev.published_at})")
    print(f"  Judul    : {ev.title}")
    print(f"  NLI      : {ev.nli_label} (Relevansi: {ev.relevance:.1%})")
    print(f"  Kutipan  : {ev.passage_text}")
```

---

## 5. Bedah Antarmuka Web App & Makna Setiap Elemen UI

Antarmuka web telah dirancang dengan standar desain tinggi (*Modern Glassmorphic Dashboard*):

```
┌────────────────────────────────────────────────────────────────────────┐
│  🔎 INDONESIAN NEWS FACT-CHECKER v2.0                                  │ ◄── Header Hero (Glassmorphic)
│  32.000 Artikel Bersih · 7 Portal Berita · mDeBERTa-v3 Multilingual    │ ◄── Status Chips Korpus
└────────────────────────────────────────────────────────────────────────┘
│                                                                        │
│  [ Input Kotak Teks Klaim                                            ] │ ◄── Tempat mengetik klaim
│  [ Contoh: Klik salah satu dari 6 tombol sampel klaim di bawah ini   ] │
│                                                                        │
│  [ ⚡ Periksa Fakta ]        [ ✖ Reset ]                                │ ◄── Tombol Aksi Utama
│                                                                        │
┌────────────────────────────────────────────────────────────────────────┐
│  VERDICT BANNER (Animasi Denyut / Pulsing Glow)                        │
│                                                                        │
│  ✔ DIDUKUNG FAKTA (SUPPORTED)                    Confidence: 98%       │ ◄── Status Verdict & Keyakinan
│  Evidence penentu secara konsisten mendukung klaim.                    │ ◄── Ringkasan Naratif
│                                                                        │
│  Distribusi Probabilitas:                                              │
│  [████████████████████░░░░░░░░░░░░░░░░░░░░] Entailment: 98.4%         │ ◄── Distribusi 3 Probabilitas
│  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Neutral: 1.2%              │
│  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Contradiction: 0.4%        │
└────────────────────────────────────────────────────────────────────────┘
│                                                                        │
│  DAFTAR BUKTI BERITA (Top Evidences):                                  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 👑 EVIDENCE PENENTU · TEMPO · 2023-03-04 · Relevansi: 62%   [↗] │  │ ◄── Spotlight Bukti Penentu
│  │ Judul: "Jokowi Perintahkan Wapres Ma'ruf Amin Tinjau Lokasi..."  │  │     (Berbingkai Emas Bercahaya)
│  │ Kutipan Berita:                                                  │  │
│  │ "Presiden Joko Widodo atau Jokowi memerintahkan Wakil Presiden..."│  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ BUKTI #2 · KOMPAS · 2023-03-05 · Relevansi: 56%             [↗] │  │ ◄── Bukti Pendukung Lainnya
│  │ ...                                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  EXPLAINABLE AI (SHAP TOKEN SALIENCY):                                 │
│  Sorotan kata pada bukti penentu yang memengaruhi keputusan:           │
│  [memerintahkan: +0.24]  [Wapres: +0.12]  [tinjau: +0.10]              │ ◄── Token Saliency Chips
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Arti Warna Indikator:
- 🟢 **Hijau Zamrud (*Emerald Green*)**: Menunjukkan hubungan **`DIDUKUNG FAKTA`** / *Entailment*. Bukti berita membuktikan bahwa klaim tersebut benar.
- 🔴 **Merah Mawar (*Rose Crimson*)**: Menunjukkan hubungan **`BERTENTANGAN / HOAKS`** / *Contradiction*. Bukti berita secara langsung membantah klaim yang diajukan.
- 🟡 **Kuning Amber (*Amber Warm*)**: Menunjukkan kondisi **`BUKTI TIDAK CUKUP`** / *Neutral*. Berita membahas topik serupa namun fakta spesifik klaim tidak disebutkan, atau klaim berada di luar cakupan korpus.

---

## 6. 10 Skenario Pengujian Nyata & Hasilnya

Untuk menguji keandalan sistem, Anda dapat menyalin (*copy-paste*) contoh klaim berikut ke dalam Web App atau CLI:

| # | Klaim Uji | Kategori | Label yang Diharapkan | Penjelasan Logika Sistem |
|---|---|---|---|---|
| **1** | `Presiden Joko Widodo telah memerintahkan Wakil Presiden Ma'ruf Amin untuk meninjau langsung lokasi kebakaran depo Pertamina di Plumpang.` | Fakta Benar | **DIDUKUNG FAKTA** | Berita Tempo & Detik mencatat perintah langsung Jokowi kepada Ma'ruf Amin pascakebakaran. |
| **2** | `Presiden Joko Widodo melarang keras Wakil Presiden Ma'ruf Amin untuk datang ke lokasi kebakaran depo Pertamina di Plumpang.` | Hoaks / Salah | **BERTENTANGAN / HOAKS** | Berita mencatat Jokowi *memerintahkan*, bukan *melarang*. Kata "melarang" menjadi kontradiksi kuat. |
| **3** | `Ketua Panpel Arema FC Abdul Haris divonis bebas dalam kasus Tragedi Kanjuruhan.` | Hoaks / Salah | **BERTENTANGAN / HOAKS** | Berita mencatat Abdul Haris divonis 1 tahun 6 bulan penjara, bukan bebas. |
| **4** | `FIFA resmi mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20 2023.` | Fakta Benar | **DIDUKUNG FAKTA** | Berita nasional akhir Maret 2023 memberitakan pembatalan status tuan rumah oleh FIFA. |
| **5** | `Pemerintah menetapkan 1 Ramadan 1444 H jatuh pada hari Kamis, 23 Maret 2023.` | Fakta Benar | **DIDUKUNG FAKTA** | Berita sidang isbat Kemenag menetapkan awal puasa 23 Maret 2023. |
| **6** | `Bencana longsor di Pulau Serasan tidak menimbulkan korban jiwa sama sekali.` | Hoaks / Salah | **BERTENTANGAN / HOAKS** | Berita mencatat puluhan korban meninggal akibat longsor di Serasan, Natuna. |
| **7** | `Pemerintah memberikan santunan uang tunai sebesar Rp 10 Miliar untuk setiap korban Depo Plumpang.` | Sebagian Benar / Beda Angka | **BUKTI TIDAK CUKUP** | Ada berita santunan, namun nominal angka 10 Miliar tidak pernah tercantum di korpus. |
| **8** | `Timnas sepak bola Indonesia dipastikan menjuarai Piala Dunia 2030 mendatang.` | Di Luar Konteks | **BUKTI TIDAK CUKUP** | Tidak ada satu pun berita di korpus yang memvalidasi prediksi masa depan tersebut. |
| **9** | `Peneliti di Indonesia menemukan fosil hidup dinosaurus jenis T-Rex di Pulau Sumba.` | Hoaks Absurd | **BUKTI TIDAK CUKUP** | Tidak ada artikel ilmiah yang mendukung, skor relevansi berada di bawah batas gerbang. |
| **10**| `Rafael Alun Trisambodo membantah memiliki safe deposit box berisi puluhan miliar rupiah.` | Fakta Berita | **DIDUKUNG FAKTA** | Berita mencatat pengakuan dan pemeriksaan Rafael Alun oleh KPK pada Maret 2023. |

---

## 7. Struktur File & Penjelasan Setiap Skrip

Berikut adalah peta struktur repositori proyek beserta fungsi setiap modul:

```text
Project UTS/
├── data/
│   ├── clean_corpus/             Folder korpus berita bersih hasil ekstraksi
│   ├── raw_news/                 Folder berita mentah per portal
│   └── benchmark.jsonl           Dataset evaluasi beranotasi emas (48 kasus uji)
│
├── cache/                        Folder penyimpanan cache (vektor embedding & indeks BM25)
│
├── fact_checker/                 PACKAGE UTAMA PYTHON (Modular)
│   ├── __init__.py               Ekspor kelas FactChecker & fungsi load_pipeline
│   ├── __main__.py               Titik masuk CLI terminal (python -m fact_checker ...)
│   ├── config.py                 Pusat konfigurasi, hyper-parameter, path, dan threshold
│   ├── schemas.py                Definisi struktur data dataclass (FactCheckResult, Evidence)
│   ├── preprocessing.py          Pembersihan boilerplate media & segmentasi kalimat
│   ├── retrieval.py              Modul BM25, Sentence-BERT, RRF fusion, & passage window
│   ├── nli_model.py              Modul inferensi mDeBERTa-v3 batch dengan softmax
│   ├── aggregation.py            Logika agregasi verdict (Max Pooling + Relevance Gate)
│   ├── explainability.py         Modul SHAP Partition Explainer & saliency visualizer
│   ├── pipeline.py               Orkestrator FactChecker yang merangkai Tahap 1 hingga 5
│   ├── app.py                    Aplikasi web Gradio (Glassmorphism UI & render_html)
│   ├── display.py                Format visualisasi terminal interaktif berbasis library rich
│   └── evaluation.py             Skrip benchmark kuantitatif (MRR, Recall, Macro-F1, E2E)
│
├── notebook_version/
│   └── Fact_Checker_Full_Project.ipynb  Notebook lengkap alur presentasi & tutorial UTS
│
├── reports/                      Hasil evaluasi kuantitatif
│   ├── evaluation_summary.md     Tabel metrik presisi, recall, F1-score lengkap
│   ├── evaluation_results.json   Data hasil prediksi per kasus uji dalam format JSON
│   └── confusion_matrices.png    Visualisasi Confusion Matrix NLI dan End-to-End
│
├── tests/                        Unit testing otomatis menggunakan pytest
├── requirements.txt              Daftar dependensi pustaka Python
├── sync_notebook.py              Skrip sinkronisasi aman antara package .py dan .ipynb
├── Laporan_UTS_NLP.md            Laporan resmi akademis proyek UTS NLP
├── PANDUAN_SISTEM.md             Panduan praktis arsitektur & penggunaan sistem (file ini)
└── README.md                     Dokumentasi ringkas beranda repositori
```

---

## 8. Pertanyaan Umum (FAQ) & Pemecahan Masalah

### Q1: Mengapa saat pertama kali dijalankan terasa cukup lama?
**Jawab**: Pada eksekusi perdana, sistem mengunduh model transformer dari Hugging Face (~860 MB untuk mDeBERTa dan ~470 MB untuk Sentence-BERT) serta menghitung embedding untuk korpus. Namun setelah cache tersimpan di folder `cache/`, program berikutnya akan terbuka hanya dalam hitungan detik.

### Q2: Apakah program ini membutuhkan koneksi internet saat berjalan?
**Jawab**: **Tidak**. Setelah model dan dataset terunduh pada eksekusi pertama, seluruh proses inferensi (Retrieval, NLI, SHAP) berjalan **100% secara lokal (*offline*)** di komputer Anda.

### Q3: Bagaimana jika muncul error `CUDA out of memory` di GPU?
**Jawab**: Anda dapat memaksa sistem berjalan di CPU atau menurunkan batas artikel dengan menambahkan parameter berikut di CLI:
```powershell
python -m fact_checker app --device cpu --max-articles 10000
```

### Q4: Mengapa klaim tentang berita tahun 2024 diprediksi "BUKTI TIDAK CUKUP"?
**Jawab**: Korpus data berita yang digunakan dalam proyek ini mencakup periode **Maret hingga April 2023** (~32.000 artikel). Oleh karena itu, sistem hanya memiliki pengetahuan mengenai peristiwa pada rentang waktu tersebut. Ini adalah karakteristik sistem berbasis korpus tertutup (*closed-corpus verification*).

### Q5: Bagaimana cara mengubah ambang batas keyakinan (*confidence threshold*)?
**Jawab**: Pada antarmuka Web App, buka menu accordion **"🎛️ Opsi Lanjutan"**, lalu geser slider **"Ambang Batas Keyakinan"** (default: `0.60`) dan **"Ambang Relevansi"** (default: `0.35`). Anda juga dapat mengubah nilai default di file [fact_checker/config.py](file:///c:/Users/IFHAL%20FAIZI/Downloads/KULIAH/SEMESTER%207/NLP/Project%20UTS/fact_checker/config.py).

---

*Disusun dengan dedikasi untuk Proyek Ujian Tengah Semester (UTS) — Natural Language Processing, Semester 7.*
