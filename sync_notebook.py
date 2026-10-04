"""
Script sinkronisasi antara fact_checker package (.py) dan notebook_version/Fact_Checker_Full_Project.ipynb.
Memperbarui sel-sel notebook agar sejalan dengan versi .py (Gradio Web App aktif & render visual HTML).
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
        source_text = "".join(cell.get("source", []))

        # 1. Update Sel 11 (Section 8: Demo End-to-End dengan HTML card)
        if "from fact_checker.display import render_result" in source_text and "fc.check" in source_text:
            cell["source"] = [
                "from fact_checker.display import render_result\n",
                "from fact_checker.app import render_html\n",
                "from IPython.display import HTML, display\n",
                "\n",
                "result = fc.check(\"Presiden Jokowi melarang Wakil Presiden Ma'ruf Amin mengunjungi lokasi kebakaran Depo Plumpang.\")\n",
                "render_result(result)\n",
                "\n",
                "# Tampilkan versi kartu visual interaktif yang identik dengan Web App\n",
                "display(HTML(render_html(result)))"
            ]
            updated_count += 1
            print(f"[OK] Sel {i+1} (Section 8 - Demo End-to-End) diperbarui dengan render_html.")

        # 2. Update Sel 21 (Section 12: Web App Gradio)
        if "build_app" in source_text and ("# import gradio" in source_text or "build_app(fc)" in source_text):
            cell["source"] = [
                "from fact_checker.app import build_app\n",
                "\n",
                "# Jalankan dashboard Web App interaktif langsung di dalam notebook\n",
                "demo = build_app(fc)\n",
                "demo.queue().launch(share=False, inbrowser=False)"
            ]
            updated_count += 1
            print(f"[OK] Sel {i+1} (Section 12 - Web App) diperbarui & diaktifkan.")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nBerhasil menyinkronkan {updated_count} sel pada {NOTEBOOK_PATH}!")

if __name__ == "__main__":
    sync()
