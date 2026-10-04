import pandas as pd

from fact_checker.preprocessing import clean_text, make_passages, preprocess_dataset, split_sentences


def test_clean_text_removes_dateline_and_footer():
    raw = 'TEMPO.CO, Jakarta - Api berhasil dipadamkan.Warga lega. Pilihan Editor: Judul lain'
    assert clean_text(raw) == "Api berhasil dipadamkan. Warga lega."


def test_clean_text_other_portals():
    assert clean_text("Jakarta, CNBC Indonesia - Harga naik.") == "Harga naik."
    assert clean_text("JAKARTA - Polisi menangkap pelaku.") == "Polisi menangkap pelaku."
    assert clean_text("Suara.com - Beredar narasi hoaks.") == "Beredar narasi hoaks."
    assert "Gambas" not in clean_text("Isi berita. [Gambas:Video CNN]")
    assert "BACA JUGA" not in clean_text("Gempa terjadi. BACA JUGA: Gempa lain")


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_split_sentences_respects_abbreviations():
    text = "Kombes Pol. Dani Hamdani mengimbau warga. Harga naik jadi Rp. 13.300 per liter. Dr. M. Nasir hadir di sana."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].startswith("Kombes Pol. Dani")
    assert "Rp. 13.300" in sents[1]
    assert sents[2] == "Dr. M. Nasir hadir di sana."


def test_make_passages_covers_whole_text():
    text = " ".join(f"Ini adalah kalimat nomor {i} dalam berita." for i in range(7))
    passages = make_passages(text, sentences_per_passage=3, stride=2)
    assert passages[0].startswith("Ini adalah kalimat nomor 0")
    assert passages[-1].endswith("nomor 6 dalam berita.")
    assert all(len(split_sentences(p)) <= 3 for p in passages)


def test_preprocess_dataset_drops_short_and_duplicates():
    long_text = "kata " * 40
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["A", "B", "C"],
        "content": [long_text, long_text, "terlalu pendek"],
        "summary": ["s", "s", "s"],
    })
    out = preprocess_dataset(df)
    assert list(out["id"]) == [1]
