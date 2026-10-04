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
              │ artikel → passage 3 kalimat        │──▶│ IndoBERT NLI / mDeBERTa-v3           │
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
| NLI & Verifikasi | `LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta` (IndoBERT NLI) & `MoritzLaurer/mDeBERTa-v3-base-xnli` |
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
| **`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (versi baru)** | **88,9%** | **0,890** |

<!-- RETRIEVAL_E2E -->

Laporan lengkap: [`reports/evaluation_summary.md`](reports/evaluation_summary.md) ·
confusion matrix: [`reports/confusion_matrices.png`](reports/confusion_matrices.png)

---

## 🚀 Cara Menjalankan

```bash
# 1. Aktifkan virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 2. Install dependensi
pip install -r requirements.txt

# 3. (Sekali saja) bangun indeks retrieval — ±10 menit di CPU, lalu di-cache di folder cache/
python -m fact_checker build-index
```

| Perintah | Fungsi |
|---|---|
| `python -m fact_checker check "klaim..."` | periksa satu klaim (`--no-explain`, `--json out.json`, `--save-plot shap.png`) |
| `python -m fact_checker interactive` | mode tanya-jawab di terminal |
| `python -m fact_checker demo` | 3 contoh klaim: benar, hoaks, di luar korpus |
| `python -m fact_checker app` | **web app Gradio** di http://127.0.0.1:7860 (`--share` untuk link publik) |
| `python -m fact_checker evaluate --compare-baseline` | evaluasi lengkap → folder `reports/` |
| `python -m pytest tests` | unit test (tanpa perlu model) |

Opsi umum: `--max-articles 5000` (hemat RAM), `--top-passages 5`, `--aggregation max|weighted|mean`,
`--device cuda`, `-v` (log detail). Skrip lama `python fact_checker/main.py` dan
`python fact_checker/run_test.py` tetap berfungsi.

Contoh pemakaian sebagai library:

```python
from fact_checker import FactChecker
fc = FactChecker()
r = fc.check("FIFA mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20 2023.")
print(r.verdict.label, f"{r.verdict.confidence:.0%}", r.verdict.reason)
for ev in r.evidences:
    print(ev.source, ev.title, ev.nli_label, f"{ev.relevance:.0%}")
```

---

## 📚 Dokumentasi Proyek

Untuk memudahkan pemahaman bagi dosen, penguji, rekan tim, maupun pengguna umum, telah disediakan dokumentasi terstruktur:

| Dokumen | Deskripsi & Tujuan |
|---|---|
| 📘 [**`LAPORAN_UTS_FACT_CHECKER_NLP.md`**](LAPORAN_UTS_FACT_CHECKER_NLP.md) | **Laporan Resmi Akademis UTS**: Latar belakang, formulasi matematis, metodologi FEVER, bedah arsitektur mDeBERTa-v3 & Hybrid RRF, evaluasi kuantitatif lengkap, dan referensi ilmiah. |
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
│   ├── nli_model.py             NLI batch dengan pemetaan label mDeBERTa-v3
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
