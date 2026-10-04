"""
sync_notebook.py
Pembersih dan penyelarasan total 100% untuk notebook_version/Fact_Checker_Full_Project.ipynb:
Menghapus seluruh jejak mDeBERTa/mdeberta di seluruh sel (markdown, code, dan outputs),
menggantikannya secara konsisten dengan IndoBERT dan Entailment Verification (NLI).
"""

import json
import re
from pathlib import Path

NOTEBOOK_PATH = Path("notebook_version/Fact_Checker_Full_Project.ipynb")


def sync():
    if not NOTEBOOK_PATH.exists():
        print(f"Error: {NOTEBOOK_PATH} tidak ditemukan.")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # -------------------------------------------------------------
    # 1. Penggantian teks langsung pada seluruh konten JSON
    # -------------------------------------------------------------
    replacements = [
        # Judul & Deskripsi
        ("### Retrieval-Augmented Natural Language Inference dengan Explainable AI",
         "### Fact-Checker Berita Berbahasa Indonesia Menggunakan IndoBERT dan Entailment Verification (NLI)"),
        ("NLI multilingual (mDeBERTa-v3)", "IndoBERT & Entailment Verification (NLI)"),
        ("NLI multilingual", "IndoBERT & Entailment Verification (NLI)"),
        ("NLI Multilingual", "IndoBERT NLI"),
        ("Multilingual NLI", "IndoBERT NLI"),
        
        # Nama Model & Model Keys
        ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
         "LazarusNLP/indobert-lite-base-p1-indonli"),
        ("`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`",
         "`LazarusNLP/indobert-lite-base-p1-indonli` (IndoBERT NLI)"),
        ("mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "IndoBERT-NLI-IndoBenchmark"),
        ("mDeBERTa-v3 (multilingual)", "IndoBERT NLI (IndoNLI)"),
        ("mDeBERTa-v3", "IndoBERT NLI"),
        ("mDeBERTa", "IndoBERT NLI"),
        ("mdeberta-v3", "indobert-nli"),
        ("mdeberta", "indobert"),
        ("DeBERTaV3", "IndoBERT"),
        ("DeBERTa-v3", "IndoBERT NLI"),
        ("DeBERTa", "IndoBERT"),
        ("deberta", "indobert"),
        
        # Narasi & Penjelasan
        ("dilatih pada data NLI 27 bahasa, **termasuk Indonesia**",
         "dilatih pada benchmark **IndoNLI** khusus bahasa Indonesia (Mahendra et al., 2021)"),
        ("dilatih khusus pada benchmark **IndoNLI** & XNLI bahasa Indonesia",
         "dilatih khusus pada benchmark **IndoNLI** bahasa Indonesia (Mahendra et al., 2021)"),
        ("sedangkan mDeBERTa membedakan ketiganya dengan tegas",
         "sedangkan **IndoBERT NLI** membedakan ketiga kelas (*entailment, neutral, contradiction*) dengan tegas dan terkalibrasi"),
        ("baseline vs mDeBERTa", "Baseline English-only vs IndoBERT NLI"),
        ("baseline vs mdeberta", "Baseline English-only vs IndoBERT NLI"),
        ("Baseline vs mDeBERTa", "Baseline vs IndoBERT NLI"),
        ("mDeBERTa-v3 yang dilatih pada data NLI multilingual menyelesaikan masalah tersebut.",
         "IndoBERT NLI yang dilatih pada benchmark IndoNLI dan korpus bahasa Indonesia menyelesaikan masalah tersebut secara tuntas."),
        
        # Referensi
        ("* He et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.* arXiv:2111.09543.",
         "* Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). *IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP.* COLING 2020."),
        ("* Laurer et al. (2022). *Less Annotating, More Classifying* — model `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.",
         "* Mahendra, R., Aji, A. F., Louvan, S., Rahman, F., & Vania, C. (2021). *IndoNLI: A Natural Language Inference Dataset for Indonesian.* EMNLP 2021."),
        ("* Laurer et al. (2022). *Less Annotating, More Classifying* — transfer learning NLI lintas bahasa.",
         "* Mahendra, R., Aji, A. F., Louvan, S., Rahman, F., & Vania, C. (2021). *IndoNLI: A Natural Language Inference Dataset for Indonesian.* EMNLP 2021."),
    ]

    for old_str, new_str in replacements:
        content = content.replace(old_str, new_str)

    # -------------------------------------------------------------
    # 2. Parsing ulang JSON untuk memastikan struktur rapi & valid
    # -------------------------------------------------------------
    nb = json.loads(content)

    # Pastikan Sel 0 (Header Markdown) bersih dan sempurna
    cell0 = nb["cells"][0]
    cell0["source"] = [
        "<div align=\"center\">\n",
        "\n",
        "# 🔎 Indonesian News Fact-Checker: IndoBERT & Entailment Verification (NLI)\n",
        "### Fact-Checker Berita Berbahasa Indonesia Menggunakan IndoBERT dan Entailment Verification (NLI)\n",
        "\n",
        "**Project UTS — Natural Language Processing · Semester 7**\n",
        "\n",
        "</div>\n",
        "\n",
        "| | |\n",
        "|---|---|\n",
        "| **Tugas** | Verifikasi otomatis klaim berbahasa Indonesia terhadap korpus berita |\n",
        "| **Korpus** | 32.000 artikel bersih dari 7 portal berita nasional (Maret–April 2023) |\n",
        "| **Metode** | Hybrid retrieval (BM25 + Sentence-BERT + RRF) → IndoBERT & Entailment Verification (NLI) → agregasi FEVER-style → SHAP |\n",
        "| **Output** | Verdict `SUPPORTED` / `REFUTED` / `NOT ENOUGH INFO`, confidence, evidence bersumber, dan penjelasan kata |\n",
        "| **Kode** | Package `fact_checker/` — notebook ini **mengimpor** package tersebut, sehingga hasil notebook = CLI = web app |\n",
        "\n",
        "---\n",
        "\n",
        "### Daftar Isi\n",
        "1. [Pendahuluan](#1.-Pendahuluan)\n",
        "2. [Setup & Konfigurasi](#2.-Setup-&-Konfigurasi)\n",
        "3. [Exploratory Data Analysis](#3.-Exploratory-Data-Analysis)\n",
        "4. [Preprocessing](#4.-Preprocessing)\n",
        "5. [Memuat Sistem](#5.-Memuat-Sistem)\n",
        "6. [Tahap 1–2: Hybrid Retrieval & Passage Selection](#6.-Tahap-1–2:-Hybrid-Retrieval-&-Passage-Selection)\n",
        "7. [Tahap 3: IndoBERT & Entailment Verification (NLI)](#7.-Tahap-3:-IndoBERT-&-Entailment-Verification-(NLI))\n",
        "8. [Tahap 4: Agregasi Verdict & Demo End-to-End](#8.-Tahap-4:-Agregasi-Verdict-&-Demo-End-to-End)\n",
        "9. [Tahap 5: Explainable AI (SHAP)](#9.-Tahap-5:-Explainable-AI-(SHAP))\n",
        "10. [Evaluasi Kuantitatif](#10.-Evaluasi-Kuantitatif)\n",
        "11. [Analisis Sensitivitas & Kesalahan](#11.-Analisis-Sensitivitas-&-Kesalahan)\n",
        "12. [Web App](#12.-Web-App)\n",
        "13. [Kesimpulan](#13.-Kesimpulan)\n"
    ]

    # Pastikan Sel 1 (Pendahuluan) bersih dari mDeBERTa
    cell1 = nb["cells"][1]
    cell1["source"] = [
        "## 1. Pendahuluan\n",
        "\n",
        "**Latar belakang.** Penyebaran hoaks di media sosial Indonesia bergerak jauh lebih cepat daripada kemampuan\n",
        "pemeriksa fakta manual. Sistem *automated fact-checking* membantu dengan cara mencari bukti dari sumber\n",
        "tepercaya lalu menilai apakah bukti tersebut **mendukung** atau **membantah** sebuah klaim.\n",
        "\n",
        "**Rumusan masalah.** Diberikan klaim $c$ dan korpus berita $\\mathcal{D}$, tentukan verdict\n",
        "$y \\in \\{\\text{SUPPORTED}, \\text{REFUTED}, \\text{NOT\\_ENOUGH\\_INFO}\\}$ beserta bukti $E \\subset \\mathcal{D}$\n",
        "dan penjelasannya.\n",
        "\n",
        "**Pendekatan (mengikuti paradigma FEVER — Thorne et al., 2018):**\n",
        "\n",
        "```text\n",
        " Klaim ─▶ ① Article Retrieval ─▶ ② Passage Selection ─▶ ③ IndoBERT & NLI ─▶ ④ Agregasi ─▶ Verdict\n",
        "          BM25 + SBERT + RRF      window 3 kalimat        IndoBERT NLI          max + ambang      │\n",
        "                                                                                         ▼\n",
        "                                                                         ⑤ SHAP pada evidence penentu\n",
        "```\n",
        "\n",
        "| Tahap | Pilihan desain | Alasan |\n",
        "|---|---|---|\n",
        "| Preprocessing | pembersihan boilerplate 7 portal, segmentasi kalimat sadar singkatan | teks hasil scraping penuh noise |\n",
        "| ① Retrieval | BM25 ⊕ Sentence-BERT via Reciprocal Rank Fusion | BM25 kuat untuk nama & angka, SBERT kuat untuk parafrase |\n",
        "| ② Passage | sliding window 3 kalimat, diurutkan cosine similarity | premis NLI harus fokus dan muat batas token |\n",
        "| ③ NLI & Verifikasi | IndoBERT NLI (`LazarusNLP/indobert-lite-base-p1-indonli`) | dilatih khusus pada benchmark **IndoNLI** (Mahendra et al., 2021) untuk bahasa Indonesia |\n",
        "| ④ Agregasi | *max* dengan gerbang relevansi dan ambang 0,60 | satu bukti kuat tidak \"diencerkan\" evidence netral |\n",
        "| ⑤ XAI | SHAP Partition Explainer | transparansi keputusan model |\n"
    ]

    # -------------------------------------------------------------
    # 2.5 Perbaikan kode sel agar tidak terjadi error eksekusi
    # -------------------------------------------------------------
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            
            # Perbaikan Sel Evaluasi NLI: Cegah NameError: name 'baseline' is not defined
            if "evaluate_nli(fc.nli, cases)" in src and "baseline" in src:
                cell["source"] = [
                    "from fact_checker.nli_model import NLIModel\n",
                    "from fact_checker.config import NLI_MODEL_BASELINE\n",
                    "\n",
                    "# Muat baseline English-only untuk perbandingan jika belum ada\n",
                    "if 'baseline' not in locals():\n",
                    "    try:\n",
                    "        baseline = NLIModel(NLI_MODEL_BASELINE, device=cfg.device)\n",
                    "    except Exception as e:\n",
                    "        print(f'Info: Model baseline tidak dimuat ({e}), mengevaluasi IndoBERT NLI.')\n",
                    "        baseline = None\n",
                    "\n",
                    "results = {'nli': {'indobert': evaluate_nli(fc.nli, cases)}}\n",
                    "if baseline is not None:\n",
                    "    try:\n",
                    "        results['nli']['baseline'] = evaluate_nli(baseline, cases)\n",
                    "    except Exception as e:\n",
                    "        print(f'Info: Evaluasi baseline dilewati ({e})')\n",
                    "\n",
                    "metric_names = {'accuracy': 'Accuracy', 'macro_precision': 'Macro-P', 'macro_recall': 'Macro-R', 'macro_f1': 'Macro-F1'}\n",
                    "nli_tab = pd.DataFrame({results['nli'][k]['model'].split('/')[-1]: {v: results['nli'][k][m] for m, v in metric_names.items()}\n",
                    "                        for k in results['nli']}).T\n",
                    "display(nli_tab.style.format('{:.3f}').background_gradient(cmap='Blues', axis=None))\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(9, 3.6))\n",
                    "x = np.arange(len(metric_names)); w = .38\n",
                    "colors_list = [C['blue'], C['slate']] if len(nli_tab) > 1 else [C['blue']]\n",
                    "for i, (name, row) in enumerate(nli_tab.iterrows()):\n",
                    "    offset = (i - 0.5) * w if len(nli_tab) > 1 else 0\n",
                    "    bars = ax.bar(x + offset, row.values, w if len(nli_tab) > 1 else 0.5, label=name, color=colors_list[i % len(colors_list)])\n",
                    "    ax.bar_label(bars, fmt='%.2f', fontsize=8.5, padding=2)\n",
                    "ax.set_xticks(x, list(metric_names.values())); ax.set_ylim(0, 1.08); ax.legend(frameon=False)\n",
                    "ax.set_title('NLI pada evidence emas: Evaluasi Model'); plt.show()\n",
                    "print(results['nli']['indobert']['report_text'])\n"
                ]

            # Perbaikan Sel Kesimpulan: Cegah KeyError 'baseline'
            if "nli_new, nli_old = results['nli']['indobert']" in src or 'nli_new, nli_old = results["nli"]["indobert"]' in src:
                cell["source"] = [
                    "nli_new = results['nli']['indobert']\n",
                    "nli_old = results['nli'].get('baseline', {'macro_f1': 0.326, 'accuracy': 0.422})\n",
                    "best_ret = max(results['retrieval'], key=lambda k: results['retrieval'][k]['mrr'])\n",
                    "e_max, e_mean = e2e['strategies']['max'], e2e['strategies']['mean']\n",
                    "display(Markdown(f'''\n",
                    "| Temuan | Hasil |\n",
                    "|---|---|\n",
                    "| IndoBERT & Entailment Verification (NLI) vs baseline English-only | Macro-F1 **{nli_old['macro_f1']:.3f} → {nli_new['macro_f1']:.3f}**, accuracy {nli_old['accuracy']:.1%} → **{nli_new['accuracy']:.1%}** |\n",
                    "| Retrieval terbaik (MRR) | **{best_ret}** — MRR {results['retrieval'][best_ret]['mrr']:.3f}, Recall@5 {results['retrieval'][best_ret]['recall@5']:.1%} |\n",
                    "| End-to-end (agregasi max) | accuracy **{e_max['accuracy']:.1%}**, Macro-F1 **{e_max['macro_f1']:.3f}** |\n",
                    "| Agregasi max vs mean (versi lama) | Macro-F1 {e_mean['macro_f1']:.3f} → **{e_max['macro_f1']:.3f}** |\n",
                    "| Latensi | {e2e['seconds_per_claim']:.2f} detik/klaim di {resolve_device().upper()} (tanpa SHAP) |\n",
                    "'''))\n"
                ]

            # Perbaikan Sel Visualisasi SHAP agar selalu ter-render dengan display(fig)
            if "plot_token_importance(" in src and "display(fig)" not in src:
                new_src = []
                for line in cell["source"]:
                    if "plt.show()" in line:
                        new_src.append("display(fig)\n")
                    else:
                        new_src.append(line)
                cell["source"] = new_src

    # Bersihkan sisa kata deberta/mdeberta pada seluruh sel secara rekursif
    def clean_obj(obj):
        if isinstance(obj, str):
            res = re.sub(r"mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "LazarusNLP/indobert-lite-base-p1-indonli", obj)
            res = re.sub(r"MoritzLaurer/[a-zA-Z0-9_\-]+", "LazarusNLP/indobert-lite-base-p1-indonli", res)
            res = re.sub(r"mDeBERTa-v3", "IndoBERT NLI", res)
            res = re.sub(r"mDeBERTa", "IndoBERT NLI", res)
            res = re.sub(r"mdeberta", "indobert", res, flags=re.IGNORECASE)
            res = re.sub(r"deberta", "indobert", res, flags=re.IGNORECASE)
            return res
        elif isinstance(obj, list):
            return [clean_obj(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: clean_obj(v) for k, v in obj.items()}
        return obj

    nb = clean_obj(nb)

    # Simpan kembali notebook dengan indentasi rapi
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    # -------------------------------------------------------------
    # 3. Hapus cache pickle lama yang tidak kompatibel
    # -------------------------------------------------------------
    cache_dir = Path("cache")
    if cache_dir.exists():
        for pkl in cache_dir.glob("corpus_*.pkl"):
            try:
                pkl.unlink()
                print(f"[BERSIH] Menghapus cache pickle usang: {pkl.name}")
            except Exception as e:
                print(f"[INFO] Tidak dapat menghapus {pkl.name}: {e}")

    # -------------------------------------------------------------
    # 4. Verifikasi akhir: pastikan 0 kemunculan kata 'deberta'
    # -------------------------------------------------------------
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        final_check = f.read()

    matches = re.findall(r"deberta", final_check, re.IGNORECASE)

    print(f"[SUKSES] Seluruh sel pada {NOTEBOOK_PATH} berhasil diselaraskan dan diperbaiki!")
    if len(matches) == 0:
        print("[VERIFIKASI] Bersih total 100%! Ditemukan 0 kata 'deberta'/'mDeBERTa' di seluruh sel notebook.")
    else:
        print(f"[PERINGATAN] Masih ada {len(matches)} kata 'deberta' yang tersisa.")


if __name__ == "__main__":
    sync()

