# 🔎 Indonesian News Fact-Checker: IndoBERT & Entailment Verification (NLI)

> **Project UTS — Natural Language Processing (Semester 7)**
> Verifikasi klaim berbahasa Indonesia terhadap **32.000 artikel berita** (7 portal, Maret–April 2023)
> menggunakan **IndoBERT dan Entailment Verification (NLI)**, *retrieval-augmented hybrid search*, dan *Explainable AI (SHAP)*.

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![torch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![hf](https://img.shields.io/badge/🤗-Transformers-yellow) ![indobert](https://img.shields.io/badge/model-IndoBERT%20%26%20NLI-green) ![tests](https://img.shields.io/badge/tests-18%20passed-brightgreen)

Diberikan sebuah klaim, sistem akan:
1. mencari artikel & passage bukti yang relevan dari korpus 32.000 berita nasional,
2. memverifikasi hubungan inferensi logika klaim–bukti menggunakan **IndoBERT & Entailment Verification** (*entailment / neutral / contradiction*),
3. memberi **verdict** — `DIDUKUNG FAKTA` · `BERTENTANGAN / HOAKS` · `BUKTI TIDAK CUKUP` — beserta tingkat keyakinan (*confidence*),
4. menjelaskan kata mana yang paling memengaruhi keputusan (Explainable AI via SHAP).

---

## 🏗️ Arsitektur

```text
              ┌───────────────────────── Tahap 1: Article Retrieval ─────────────────────────┐
  Klaim ──┬──▶│ BM25 (judul + isi)          ─┐                                               │
          │   │                               ├─▶ Reciprocal Rank Fusion ─▶ Top-8 artikel     │
          └──▶│ Sentence-BERT (judul+ringkasan)┘   RRF(d) = Σ 1/(60 + rank)                   │
              └───────────────────────────────────────────────┬───────────────────────────────┘
                                                              ▼
              ┌──── Tahap 2: Passage Selection ────┐   ┌──── Tahap 3: IndoBERT & Entailment ──┐
              │ artikel → passage 3 kalimat        │──▶│ IndoBERT NLI (IndoBenchmark)         │
              │ (sliding window), urut cosine sim  │   │ premise = passage, hypothesis = klaim│
              └────────────────────────────────────┘   └──────────────────┬───────────────────┘
                                                                          ▼
              ┌──── Tahap 5: Explainability ───────┐   ┌──── Tahap 4: Agregasi (FEVER-style) ─┐
              │ SHAP Partition Explainer pada      │◀──│ gerbang relevansi ≥ 0,35             │
              │ evidence penentu                   │   │ SUPPORTED bila P(E) ≥ 0,60           │
              └────────────────────────────────────┘   │ REFUTED   bila P(C) ≥ 0,60, else NEI │
                                                       └──────────────────────────────────────┘
```

| Komponen | Teknologi |
|---|---|
| Preprocessing | pembersihan boilerplate 7 portal, segmentasi kalimat sadar singkatan Indonesia |
| Sparse retrieval | Okapi BM25 (implementasi vektor, matriks sparse SciPy) |
| Dense retrieval | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Fusion | Reciprocal Rank Fusion (Cormack et al., 2009) |
| NLI & Verifikasi | `LazarusNLP/indobert-lite-base-p1-indonli` (IndoBERT NLI ter-fine-tune benchmark IndoNLI) |
| Explainability | SHAP (Partition Explainer) + fallback occlusion |
| Antarmuka | CLI (`rich`), Web App (`gradio`), Jupyter Notebook |

---

## 📊 Hasil Evaluasi

Benchmark `data/benchmark.jsonl`: **48 kasus** — 15 topik berita × 3 label dengan evidence emas yang
diambil langsung dari `data.csv`, ditambah 3 klaim di luar cakupan korpus.

**NLI dengan evidence emas (45 pasangan)**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| `cross-encoder/nli-MiniLM2-L6-H768` (versi lama, English-only) | 42,2% | 0,326 |
| **`IndoBERT NLI / Multilingual Verifier` (versi baru)** | **88,9%** | **0,890** |

<!-- RETRIEVAL_E2E -->

Laporan lengkap: [`reports/evaluation_summary.md`](reports/evaluation_summary.md) ·
confusion matrix: [`reports/confusion_matrices.png`](reports/confusion_matrices.png)

---

## 🚀 Cara Menjalankan Project

Pilih salah satu cara menjalankan yang paling sesuai dengan kebutuhan Anda: **Versi Python (.py)** melalui Web App/CLI atau **Versi Jupyter Notebook (.ipynb)** untuk eksplorasi sel-per-sel.

---

### ⚙️ Persiapan Awal Lingkungan (Dilakukan Sekali Saja)

Pastikan virtual environment telah aktif dan dependensi terpasang:

```powershell
# 1. Buka PowerShell di folder project dan aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# 2. Instalasi dependensi (jika belum terpasang)
pip install -r requirements.txt

# 3. Bangun indeks retrieval 32.000 berita (hanya 1x di awal, di-cache ke folder cache/)
python -m fact_checker build-index
```
> [!TIP]
> Pembangunan indeks membutuhkan waktu ~8-10 menit di CPU. Setelah tersimpan di folder `cache/`, pemuatan indeks pada eksekusi berikutnya berlangsung instan (**hanya ~2 detik**).

---

### 🅰️ Pilihan 1: Menjalankan Program Python (`.py`)

#### 1. Web App Interaktif Modern (Gradio Dashboard) — *Sangat Direkomendasikan*
Antarmuka visual modern dengan kartu bukti, probabilitas, dan visualisasi kata SHAP:
```powershell
# Jalankan server lokal
python -m fact_checker app

# Atau aktifkan tautan publik (bisa diakses siapa saja via internet selama 72 jam):
python -m fact_checker app --share
```
👉 Buka di browser Anda: **`http://127.0.0.1:7860`**

#### 2. Terminal CLI (Cepat & Ringkas Tanpa Browser)
```powershell
# A. Periksa satu klaim langsung:
python -m fact_checker check "Presiden Jokowi meresmikan Jalan Tol Trans Sumatera."

# B. Mode Tanya-Jawab Interaktif di terminal:
python -m fact_checker interactive

# C. Mode Demo Otomatis (menguji 3 skenario: didukung, hoaks, di luar korpus):
python -m fact_checker demo

# D. Menjalankan Evaluasi Benchmark 48 Kasus Uji:
python -m fact_checker evaluate --compare-baseline

# E. Menjalankan Unit Test (pytest):
python -m pytest tests
```

#### 3. Mengimpor sebagai Library Python pada Skrip Sendiri
Buat file skrip Python (misal `test_cekklaim.py`) dan panggil engine `fact_checker`:
```python
from fact_checker import FactChecker

# Inisialisasi engine fact-checker
fc = FactChecker()

# Uji sebuah klaim
result = fc.check("FIFA membatalkan status Indonesia sebagai tuan rumah Piala Dunia U-20 2023.")

# Cetak hasil
print(f"Verdict    : {result.verdict.label}")
print(f"Confidence : {result.verdict.confidence:.1%}")
print(f"Alasan     : {result.verdict.reason}")
for i, ev in enumerate(result.evidences, 1):
    penentu = " (👑 PENENTU)" if ev.is_decisive else ""
    print(f"Bukti #{i}{penentu}: [{ev.source}] {ev.title} -> {ev.nli_label} ({ev.relevance:.1%})")
```
Jalankan skrip:
```powershell
python test_cekklaim.py
```

---

### 🅱️ Pilihan 2: Menjalankan Jupyter Notebook (`.ipynb`)

File notebook terletak di: [`notebook_version/Fact_Checker_Full_Project.ipynb`](notebook_version/Fact_Checker_Full_Project.ipynb)

Notebook ini berisi alur akademis menyeluruh: **EDA → Preprocessing → Hybrid Retrieval (BM25 + SBERT + RRF) → IndoBERT NLI → Agregasi FEVER → SHAP XAI → Evaluasi Kuantitatif → Web App tersemat**.

#### Cara 1: Menggunakan VS Code (Paling Praktis)

> [!IMPORTANT]
> **WAJIB MENGGUNAKAN PYTHON 3.10 UNTUK IPYNB!**  
> Seluruh pustaka (PyTorch, Transformers, Sentence-Transformers, Gradio, SHAP, Scikit-learn, Pandas) dipasang secara khusus dan stabil di dalam virtual environment Python 3.10 (`venv`). Jangan menggunakan Python 3.12 atau versi Python global lainnya.

**Langkah Memilih Kernel di VS Code:**
1. Buka file [`notebook_version/Fact_Checker_Full_Project.ipynb`](notebook_version/Fact_Checker_Full_Project.ipynb) di VS Code.
2. Di pojok kanan atas jendela notebook, klik **indikator kernel** (misalnya tertulis `Python 3.12` atau `Select Kernel`).
3. Pilih **Python Environments...** → pilih kernel: **`Python 3.10.x ('venv': venv)`** (virtual environment proyek yang memiliki seluruh dependensi terpasang lengkap).
4. Klik tombol **Restart Kernel** (ikon putar balik melingkar) lalu jalankan sel dari awal atau klik **Run All** (atau `Shift + Enter` per sel). Sel 7 kini akan memuat data secara lancar tanpa error `NotImplementedError`!
5. Di **Bagian 8**, hasil verifikasi akan langsung muncul dalam bentuk tabel terminal dan **kartu visual HTML**.
6. Di **Bagian 12**, Web App Gradio akan langsung aktif dan interaktif di dalam output sel notebook!

#### Cara 2: Menggunakan Jupyter Lab / Notebook Browser
Jika Anda lebih terbiasa dengan antarmuka web Jupyter:
```powershell
# 1. Pastikan jupyter terpasang & buka server jupyter
jupyter notebook
# atau: jupyter lab
```
1. Browser akan terbuka otomatis di `http://localhost:8888`.
2. Navigasikan ke folder `notebook_version/` lalu klik `Fact_Checker_Full_Project.ipynb`.
3. Pastikan kernel yang dipilih adalah kernel virtual environment `venv`.
4. Jalankan sel secara berurutan (`Shift + Enter`).

---

## 📚 Dokumentasi Proyek

Untuk memudahkan pemahaman bagi dosen, penguji, rekan tim, maupun pengguna umum, telah disediakan dokumentasi terstruktur:

| Dokumen | Deskripsi & Tujuan |
|---|---|
| 🚀 [**`PANDUAN_MENJALANKAN.md`**](PANDUAN_MENJALANKAN.md) | **Panduan Menjalankan (.py & .ipynb)**: Tutorial langkah demi langkah menjalankan Web App Gradio, Terminal CLI, dan Jupyter Notebook di VS Code / Jupyter Lab. |
| 📘 [**`LAPORAN_UTS_FACT_CHECKER_NLP.md`**](LAPORAN_UTS_FACT_CHECKER_NLP.md) | **Laporan Resmi Akademis UTS**: Latar belakang, formulasi matematis, metodologi FEVER, bedah arsitektur IndoBERT & Entailment Verification (NLI) & Hybrid RRF, evaluasi kuantitatif lengkap, dan referensi ilmiah. |
| 📖 [**`PANDUAN_LENGKAP_FACT_CHECKER.md`**](PANDUAN_LENGKAP_FACT_CHECKER.md) | **Panduan Lengkap Praktis**: Alur kerja hulu-ke-hilir, panduan 4 mode eksekusi, bedah antarmuka Web App, 10 skenario uji nyata, dan FAQ troubleshooting. |
| 📝 [**`PANDUAN_PENGUJIAN_KLAIM.md`**](PANDUAN_PENGUJIAN_KLAIM.md) | **Koleksi Klaim Pengujian**: Daftar klaim fakta benar, hoaks/salah, dan netral yang siap disalin (*copy-paste*) lengkap dengan konteks beritanya. |
| 📊 [**`reports/evaluation_summary.md`**](reports/evaluation_summary.md) | **Rangkuman Hasil Evaluasi**: Tabel metrik presisi, recall, macro-F1, dan analisis 16 kesalahan prediksi pada 48 kasus benchmark. |

---

## 📁 Struktur Project

```text
Project UTS/
├── data.csv                     dataset berita (32.000 artikel, tidak di-commit)
├── data/benchmark.jsonl         benchmark evaluasi berlabel (48 kasus uji)
├── fact_checker/                package utama (modular)
│   ├── config.py                seluruh hyper-parameter & nama model
│   ├── preprocessing.py         cleaning, segmentasi kalimat, passage
│   ├── retrieval.py             BM25, dense encoder + cache, RRF, passage selection
│   ├── nli_model.py             Modul inferensi IndoBERT & Entailment Verification (NLI)
│   ├── aggregation.py           verdict multi-evidence (FEVER-style max + gate)
│   ├── explainability.py        SHAP partition explainer, highlight HTML, plot
│   ├── pipeline.py              orkestrator FactChecker
│   ├── evaluation.py            metrik NLI, retrieval, end-to-end + laporan
│   ├── display.py               tampilan terminal (rich)
│   ├── app.py                   web app Gradio modern (glassmorphism UI & render_html)
│   └── __main__.py              CLI entrypoint
├── tests/                       unit test otomatis (pytest)
├── notebook_version/            notebook presentasi (mengimpor package)
│   └── Fact_Checker_Full_Project.ipynb
├── reports/                     hasil evaluasi & grafik
│   ├── evaluation_summary.md
│   ├── evaluation_results.json
│   └── confusion_matrices.png
├── PANDUAN_MENJALANKAN.md                panduan ringkas eksekusi .py dan .ipynb
├── LAPORAN_UTS_FACT_CHECKER_NLP.md       laporan resmi akademis UTS NLP
├── PANDUAN_LENGKAP_FACT_CHECKER.md       panduan praktis & arsitektur sistem
├── PANDUAN_PENGUJIAN_KLAIM.md           daftar contoh klaim uji
├── sync_notebook.py                      skrip sinkronisasi aman .py ke .ipynb
└── README.md                             dokumentasi ringkas beranda
```

---

## ⚠️ Keterbatasan

* Korpus hanya mencakup berita **Maret–April 2023**; klaim di luar periode ini cenderung *Bukti Tidak Cukup*.
* "Didukung berita" ≠ "pasti benar" — sistem memeriksa konsistensi klaim terhadap korpus, bukan kebenaran absolut.
* NLI masih bisa keliru pada angka, negasi ganda, dan penalaran temporal yang kompleks.
* Benchmark saat ini berjumlah 48 kasus beranotasi emas; evaluasi lebih lanjut pada dataset eksternal (seperti IndoNLI) direkomendasikan.
