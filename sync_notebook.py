"""
sync_notebook.py
Script untuk menyinkronkan seluruh konten notebook_version/Fact_Checker_Full_Project.ipynb
dengan arsitektur IndoBERT dan Entailment Verification (NLI).
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebook_version/Fact_Checker_Full_Project.ipynb")


def sync():
    if not NOTEBOOK_PATH.exists():
        print(f"Error: {NOTEBOOK_PATH} tidak ditemukan.")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated_count = 0

    for i, cell in enumerate(nb.get("cells", [])):
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)
        modified = False

        # 1. Judul Utama Notebook
        if cell.get("cell_type") == "markdown" and "# 🔎 Indonesian News Fact-Checker" in source_text:
            new_lines = []
            for line in source_lines:
                if line.strip() == "# 🔎 Indonesian News Fact-Checker":
                    new_lines.append("# 🔎 Indonesian News Fact-Checker: IndoBERT & Entailment Verification (NLI)\n")
                    modified = True
                elif "Verifikasi klaim berita berbahasa Indonesia" in line:
                    new_lines.append(
                        "> Verifikasi klaim berita berbahasa Indonesia terhadap **32.000 artikel berita** (7 portal nasional, Maret–April 2023) menggunakan **IndoBERT dan Entailment Verification (NLI)**, *retrieval-augmented hybrid search*, dan *Explainable AI (SHAP)*.\n"
                    )
                    modified = True
                elif "NLI multilingual (mDeBERTa-v3)" in line:
                    new_lines.append(
                        line.replace(
                            "NLI multilingual (mDeBERTa-v3)",
                            "IndoBERT & Entailment Verification (NLI)",
                        )
                    )
                    modified = True
                else:
                    new_lines.append(line)
            if modified:
                cell["source"] = new_lines
                updated_count += 1
                print(f"[OK] Sel {i+1} (Judul Utama & Ikhtisar) diperbarui.")

        # 2. Ringkasan Arsitektur & Tabel Komponen
        elif cell.get("cell_type") == "markdown" and "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" in source_text:
            new_lines = []
            for line in source_lines:
                if "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" in line and "③ NLI" in line:
                    new_lines.append(
                        "| ③ NLI & Verifikasi | IndoBERT NLI (`LazarusNLP/indobert-lite-base-p1-indonli`) & `mDeBERTa-v3` | dilatih khusus pada benchmark **IndoNLI** & XNLI bahasa Indonesia |\n"
                    )
                    modified = True
                elif "mDeBERTa" in line and "window 3 kalimat" in line:
                    new_lines.append(
                        line.replace("mDeBERTa", "IndoBERT/mDeBERTa")
                    )
                    modified = True
                else:
                    new_lines.append(line)
            if modified:
                cell["source"] = new_lines
                updated_count += 1
                print(f"[OK] Sel {i+1} (Tabel Komponen Arsitektur) diperbarui.")

        # 3. Penjelasan Bagian NLI (Section 5)
        elif cell.get("cell_type") == "markdown" and ("Bagian 5: Natural Language Inference" in source_text or "Bagian 5" in source_text and "NLI" in source_text):
            new_lines = []
            for line in source_lines:
                if "Bagian 5:" in line:
                    new_lines.append("## Bagian 5: IndoBERT & Entailment Verification (NLI)\n")
                    modified = True
                elif "Model yang digunakan adalah" in line:
                    new_lines.append(
                        "Model utama yang digunakan adalah **IndoBERT NLI** (`LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta`) yang di-*fine-tune* khusus pada benchmark **IndoNLI** (Mahendra et al., 2021) serta arsitektur Cross-Lingual NLI (`mDeBERTa-v3`) untuk inferensi entitas global.\n"
                    )
                    modified = True
                else:
                    new_lines.append(line)
            if modified:
                cell["source"] = new_lines
                updated_count += 1
                print(f"[OK] Sel {i+1} (Bagian 5 NLI Explanation) diperbarui.")

        # 4. Update Sel 11 (Section 8: Demo End-to-End dengan HTML card)
        elif "from fact_checker.display import render_result" in source_text and "fc.check" in source_text:
            cell["source"] = [
                "from fact_checker.display import render_result\n",
                "from fact_checker.app import render_html\n",
                "from IPython.display import HTML, display\n",
                "\n",
                "# Uji inferensi klaim menggunakan IndoBERT dan Entailment Verification\n",
                "result = fc.check(\"Presiden Jokowi melarang Wakil Presiden Ma'ruf Amin mengunjungi lokasi kebakaran Depo Plumpang.\")\n",
                "render_result(result)\n",
                "\n",
                "# Tampilkan versi kartu visual interaktif yang identik dengan Web App\n",
                "display(HTML(render_html(result)))\n"
            ]
            updated_count += 1
            print(f"[OK] Sel {i+1} (Section 8 - Demo End-to-End) diperbarui dengan render_html.")

        # 5. Update Sel Web App Gradio
        elif "build_app" in source_text and ("# import gradio" in source_text or "build_app(fc)" in source_text):
            cell["source"] = [
                "from fact_checker.app import build_app\n",
                "\n",
                "# Jalankan dashboard Web App interaktif berbasis IndoBERT & Entailment Verification langsung di dalam notebook\n",
                "demo = build_app(fc)\n",
                "demo.queue().launch(share=False, inbrowser=False)\n"
            ]
            updated_count += 1
            print(f"[OK] Sel {i+1} (Section 12 - Web App) diperbarui & diaktifkan.")

        # 6. Referensi Ilmiah di akhir notebook
        elif cell.get("cell_type") == "markdown" and ("Laurer et al." in source_text or "Thorne et al." in source_text) and "Koto et al." not in source_text:
            new_lines = list(source_lines)
            new_lines.append("\n* Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). *IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP.* In COLING 2020.\n")
            new_lines.append("* Mahendra, R., Aji, A. F., Louvan, S., Rahman, F., & Vania, C. (2021). *IndoNLI: A Natural Language Inference Dataset for Indonesian.* In EMNLP 2021.\n")
            cell["source"] = new_lines
            updated_count += 1
            print(f"[OK] Sel {i+1} (Referensi Ilmiah IndoBERT & IndoNLI) ditambahkan.")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[SELESAI] Berhasil menyinkronkan {updated_count} sel pada {NOTEBOOK_PATH}!")


if __name__ == "__main__":
    sync()
