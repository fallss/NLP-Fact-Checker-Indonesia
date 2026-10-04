# 📖 Panduan Lengkap Menjalankan Indonesian News Fact-Checker

> **Project UTS — Natural Language Processing (Semester 7)**  
> Sistem Verifikasi Klaim Berita Berbahasa Indonesia Menggunakan **IndoBERT dan Entailment Verification (NLI)**, *Hybrid Retrieval (BM25 + SBERT + RRF)*, dan *Explainable AI (SHAP)*.

---

## 📌 Daftar Pilihan Eksekusi

Anda dapat menjalankan proyek ini menggunakan dua jalur utama sesuai kebutuhan Anda:

| Opsi | Format | Antarmuka / Lingkungan | Kebutuhan Penggunaan |
|---|---|---|---|
| **Pilihan 🅰️** | **Script Python (`.py`)** | Modern Web App (Gradio), Terminal CLI, Library Script | Demonstrasi interaktif, pengujian cepat, produksi, dan presentasi dosen |
| **Pilihan 🅱️** | **Jupyter Notebook (`.ipynb`)** | VS Code Notebook, Jupyter Lab, Google Colab | Pembelajaran sel-demi-sel, eksperimen akademis, EDA, dan visualisasi grafik |

---

## ⚙️ Langkah 0: Persiapan Lingkungan (Sekali Saja)

Sebelum menjalankan opsi mana pun, pastikan Anda berada di direktori project dan virtual environment telah aktif:

### 1. Buka Terminal & Aktifkan Virtual Environment
Buka terminal PowerShell pada folder project:
```powershell
.\venv\Scripts\Activate.ps1
```
*(Pastikan muncul tanda `(venv)` di sebelah kiri prompt terminal).*

### 2. Pasang Dependensi (Jika Belum)
```powershell
pip install -r requirements.txt
```

### 3. Bangun Indeks Retrieval (Hanya Sekali di Awal)
Sistem membutuhkan indeks vektor pencarian dari 32.000 artikel berita agar pencarian bukti berjalan cepat:
```powershell
python -m fact_checker build-index
```
> [!NOTE]
> Proses ini memakan waktu sekitar **8–10 menit di CPU** (atau ~2 menit jika menggunakan GPU). Setelah selesai, indeks disimpan di dalam folder `cache/`. Pada eksekusi berikutnya, pemuatan indeks hanya memakan waktu **2 detik**!

---

## 🅰️ Panduan Menjalankan Versi Python (`.py`)

### 1. Menjalankan Modern Web App (Gradio Dashboard) — *Paling Direkomendasikan*

Antarmuka web interaktif dengan desain Glassmorphism modern, kartu bukti berbobot probabilitas, dan penyorotan kata penjelas SHAP.

```powershell
# Menjalankan server web lokal
python -m fact_checker app
```
- Buka browser web Anda di: **`http://127.0.0.1:7860`**
- **Fitur di Web App**:
  - Masukkan klaim bebas di kolom teks atau klik tombol contoh klaim yang disediakan.
  - Tentukan jumlah bukti yang dicari (default: 5 bukti).
  - Tentukan ambang batas keyakinan (*threshold*).
  - Tekan tombol **"Periksa Fakta Sekarang"**.
  - Lihat verdict berkilau (*SUPPORTED*, *REFUTED*, atau *NOT ENOUGH INFO*), kartu bukti asli bersumber berita, dan tabel penjelasan kata penentu.

#### Opsi Berbagi Tautan Publik (Public Link):
Jika ingin membagikan demo web app ke dosen atau teman melalui internet tanpa perlu mereka menginstal apa pun:
```powershell
python -m fact_checker app --share
```
*Gradio akan menghasilkan URL publik sementara (misal: `https://xxxx.gradio.live`) yang aktif selama 72 jam.*

---

### 2. Menjalankan Terminal CLI (Command-Line Interface)

Sangat praktis untuk pengujian kilat tanpa perlu membuka browser.

#### A. Periksa Satu Klaim Langsung
```powershell
python -m fact_checker check "Presiden Jokowi memerintahkan Wapres Ma'ruf Amin meninjau lokasi kebakaran Depo Plumpang."
```

#### B. Mode Tanya-Jawab Interaktif (Looping Terminal)
Masuk ke sesi tanya-jawab interaktif di mana Anda dapat mengetikkan klaim berulang kali:
```powershell
python -m fact_checker interactive
```
*(Ketik `keluar` atau `exit` untuk mengakhiri sesi).*

#### C. Mode Demo Otomatis (3 Kasus Uji Ekstrem)
Menjalankan otomatis 3 contoh klaim: fakta didukung, hoaks/kontradiksi, dan klaim di luar jangkauan berita:
```powershell
python -m fact_checker demo
```

#### D. Menjalankan Evaluasi Benchmark Lengkap
Menguji performa sistem terhadap 48 kasus uji berlabel emas (`data/benchmark.jsonl`):
```powershell
python -m fact_checker evaluate --compare-baseline
```
*Hasil metrik (Akurasi, Macro-F1, Precision, Recall) dan matriks kebingungan akan tersimpan di folder `reports/`.*

#### E. Menjalankan Unit Test Otomatis
```powershell
python -m pytest tests
```

---

### 3. Mengimpor sebagai Library Python pada Skrip Mandiri

Anda dapat memanggil engine `fact_checker` di dalam skrip Python buatan Anda sendiri:

Buat file baru, misalnya `cek_berita.py`:
```python
from fact_checker import FactChecker

# 1. Inisialisasi engine
checker = FactChecker()

# 2. Ajukan klaim
klaim = "FIFA mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20 2023."
hasil = checker.check(klaim)

# 3. Tampilkan hasil
print("=" * 60)
print(f"KLAIM     : {klaim}")
print(f"VERDICT   : {hasil.verdict.label}")
print(f"CONFIDENCE: {hasil.verdict.confidence:.1%}")
print(f"ALASAN    : {hasil.verdict.reason}")
print("=" * 60)

print("\nBUKTI YANG DITEMUKAN:")
for i, ev in enumerate(hasil.evidences, 1):
    bintang = " (👑 BUKTI PENENTU)" if ev.is_decisive else ""
    print(f"\n[{i}] {ev.source} - {ev.title}{bintang}")
    print(f"    Status NLI: {ev.nli_label} (Relevansi: {ev.relevance:.1%})")
    print(f"    Kutipan   : \"{ev.passage_text}\"")
```

Jalankan skrip:
```powershell
python cek_berita.py
```

---

## 🅱️ Panduan Menjalankan Versi Jupyter Notebook (`.ipynb`)

File notebook berada di lokasi:  
📁 [`notebook_version/Fact_Checker_Full_Project.ipynb`](notebook_version/Fact_Checker_Full_Project.ipynb)

Notebook ini dirancang sistematis mencakup 13 bagian:
1. **Pendahuluan & Teori FEVER**
2. **Setup & Konfigurasi**
3. **Exploratory Data Analysis (EDA) Korpus Berita**
4. **Preprocessing & Pembersihan Teks 7 Portal**
5. **Memuat Pipeline Sistem**
6. **Tahap 1–2: Hybrid Retrieval & Passage Selection**
7. **Tahap 3: IndoBERT & Entailment Verification (NLI)**
8. **Tahap 4: Agregasi Verdict & Demo End-to-End (Tampilan Kartu HTML)**
9. **Tahap 5: Explainable AI (SHAP)**
10. **Evaluasi Kuantitatif Benchmark (48 Kasus)**
11. **Analisis Sensitivitas Ambang Batas & Kasus Salah**
12. **Web App Gradio Tersemat di Notebook**
13. **Kesimpulan & Referensi Ilmiah**

---

### Cara 1: Menggunakan Visual Studio Code (Sangat Disarankan)

> [!IMPORTANT]
> **WAJIB MENGGUNAKAN PYTHON 3.10 UNTUK IPYNB!**  
> Seluruh pustaka (PyTorch, Transformers, Sentence-Transformers, Gradio, SHAP, Scikit-learn, Pandas) dipasang secara khusus dan stabil di dalam virtual environment Python 3.10 (`venv`). Jangan menggunakan Python 3.12 atau versi Python global lainnya karena dapat menyebabkan ketidakcocokan dependensi dan error `NotImplementedError` saat deserialisasi data.

#### Langkah Memilih Kernel di VS Code:
Ketika membuka [`notebook_version/Fact_Checker_Full_Project.ipynb`](notebook_version/Fact_Checker_Full_Project.ipynb) di VS Code:

1. **Buka Project di VS Code**:
   - Buka folder `Project UTS` di VS Code, lalu klik file `notebook_version/Fact_Checker_Full_Project.ipynb` di panel Explorer.
2. **Pilih Kernel Virtual Environment**:
   - Klik indikator kernel di **pojok kanan atas** editor notebook (misalnya tertulis `Python 3.12` atau `Select Kernel`).
   - Pilih **Python Environments...**
   - Pilih kernel: **`Python 3.10.x ('venv': venv)`** (virtual environment proyek yang memiliki seluruh dependensi terpasang lengkap).
3. **Restart Kernel & Jalankan**:
   - Klik tombol **Restart Kernel** (ikon putar balik melingkar) lalu jalankan sel dari awal atau klik **Run All** (atau gunakan shortcut `Shift + Enter`).
   - **Hasil**: Sel 7 kini akan memuat data secara lancar tanpa error `NotImplementedError`!
4. **Melihat Hasil Interaktif**:
   - Di **Bagian 8**, kartu verifikasi HTML modern akan langsung ter-render di dalam output sel.
   - Di **Bagian 12**, antarmuka Web App Gradio akan langsung hidup dan dapat digunakan langsung di dalam notebook.

---

### Cara 2: Menggunakan Jupyter Lab / Jupyter Notebook Browser

Jika Anda lebih memilih antarmuka web Jupyter:

1. **Jalankan Server Jupyter dari Terminal**:
   ```powershell
   jupyter notebook
   # atau jika menggunakan jupyter lab:
   jupyter lab
   ```
2. **Akses di Browser**:
   - Browser akan otomatis terbuka di `http://localhost:8888`.
3. **Buka Notebook**:
   - Masuk ke folder `notebook_version/` dan klik `Fact_Checker_Full_Project.ipynb`.
4. **Pastikan Kernel Tepat**:
   - Periksa menu **Kernel** → **Change Kernel** → pilih kernel Python environment `venv`.
5. **Jalankan Sel Berurutan**:
   - Gunakan shortcut `Shift + Enter` untuk mengeksekusi sel satu per satu dari atas ke bawah.

---

## 💡 Daftar Contoh Klaim untuk Diuji Coba

Berikut adalah beberapa contoh klaim yang dapat langsung Anda salin untuk menguji sistem (baik di Web App, CLI, maupun Notebook):

| Kategori | Contoh Klaim | Ekspektasi Verdict |
|---|---|---|
| **Fakta Didukung** | *"Presiden Jokowi meninjau langsung lokasi terdampak kebakaran Depo Pertamina Plumpang di Jakarta Utara."* | `DIDUKUNG FAKTA` (SUPPORTED) |
| **Fakta Didukung** | *"FIFA resmi membatalkan status Indonesia sebagai tuan rumah Piala Dunia U-20 2023."* | `DIDUKUNG FAKTA` (SUPPORTED) |
| **Fakta Didukung** | *"KPK melakukan operasi tangkap tangan terhadap Bupati Kepulauan Meranti Muhammad Adil."* | `DIDUKUNG FAKTA` (SUPPORTED) |
| **Hoaks / Kontradiksi** | *"Piala Dunia U-20 2023 tetap resmi diselenggarakan di Indonesia setelah negosiasi PSSI berhasil."* | `BERTENTANGAN / HOAKS` (REFUTED) |
| **Hoaks / Kontradiksi** | *"Presiden Joko Widodo menolak memberikan bantuan kepada para korban kebakaran Depo Plumpang."* | `BERTENTANGAN / HOAKS` (REFUTED) |
| **Bukti Tidak Cukup** | *"Elon Musk meresmikan kantor pusat SpaceX cabang Bandung pada bulan Maret 2023."* | `BUKTI TIDAK CUKUP` (NOT ENOUGH INFO) |

---

## 🛠️ Tanya-Jawab & Troubleshooting (FAQ)

### Q: Port 7860 sudah terpakai saat menjalankan `python -m fact_checker app`?
Gradio akan otomatis mencoba port berikutnya (seperti `7861`, `7862`). Perhatikan URL yang tercetak di baris log terminal.

### Q: Apakah wajib memiliki GPU NVIDIA?
Tidak. Seluruh arsitektur (BM25, Sentence-BERT, IndoBERT NLI, dan SHAP) telah dioptimasi dengan baik untuk dapat berjalan di **CPU biasa (RAM min 8 GB)**.

### Q: Mengapa eksekusi pertama terasa agak memakan waktu?
Pada eksekusi pertama, model IndoBERT NLI (`LazarusNLP/indobert-lite-base-p1-indonli`) dan model Sentence-BERT akan diunduh dari Hugging Face ke cache lokal Anda. Eksekusi berikutnya akan memuat model secara lokal dan cepat.
