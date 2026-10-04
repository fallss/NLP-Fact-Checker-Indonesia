# =============================================================================
# preprocessing.py
# Memuat, membersihkan, dan memecah (chunking) korpus berita berbahasa Indonesia.
#
# Tahapan:
#   1. load_articles()      : baca kolom yang diperlukan dari data.csv
#   2. clean_text()         : hapus boilerplate portal berita (dateline, "Baca juga",
#                             "Pilihan Editor", tag video, e-mail tersamar, dsb.)
#   3. split_sentences()    : segmentasi kalimat yang sadar singkatan bahasa Indonesia
#   4. make_passages()      : sliding window beberapa kalimat -> passage evidence
# =============================================================================

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Kolom yang dipakai. Kolom 'embedding' (vektor 1536-d bawaan dataset) sengaja
# tidak dimuat: ukurannya ~700 MB dan dibuat oleh model yang tidak tersedia offline.
USECOLS = ["id", "source", "title", "url", "date", "content", "summary"]

# -----------------------------------------------------------------------------
# Pola boilerplate hasil scraping portal berita
# -----------------------------------------------------------------------------
_DATELINE_PATTERNS = [
    r"^TEMPO\.CO\s*,\s*[A-Za-z .]{2,30}?\s*[-–—]\s*",          # TEMPO.CO, Jakarta -
    r"^[A-Za-z .]{2,30},\s*CNBC\s*Indonesia\s*[-–—]\s*",         # Jakarta, CNBC Indonesia -
    r"^Suara\.com\s*[-–—]\s*",                                   # Suara.com -
    r"^[A-Z]{3,}(?:\s[A-Z]{3,}){0,2}\s*[-–—]\s*",                # JAKARTA -  (okezone)
    r"^[A-Za-z]+\.(?:com|co|id)\s*[-–—]\s*",                     # Liputan6.com - dsb.
]
_DATELINE_RE = [re.compile(p) for p in _DATELINE_PATTERNS]

_NOISE_RE = [
    re.compile(r"Pilihan Editor\s*:.*$", re.S),                  # footer Tempo
    re.compile(r"Ikuti berita terkini dari Tempo.*$", re.S),
    re.compile(r"Simak (?:informasi selengkapnya dalam )?video di atas.*$", re.S | re.I),
    re.compile(r"\[Gambas:[^\]]*\]"),                            # embed video CNN/CNBC
    re.compile(r"\[\s*email\s*protected\s*\]", re.I),            # e-mail tersamar
    re.compile(r"Baca juga\s*:?\s*", re.I),                     # penanda tautan terkait
    re.compile(r"(?:^|\s)#\w+"),                                 # hashtag
    re.compile(r"\s*\((?:[a-z]{2,4}|[A-Z]{2,4})\)\s*$"),         # kode penulis: (wal)
]

# Kalimat yang "menempel": 'tewas."Presiden' atau 'Maret.Rofik' -> beri spasi
_GLUED_SENT_RE = re.compile(r"([a-z0-9%)][.!?][\"”’]?)(?=[\"“]?[A-Z])")
_WS_RE = re.compile(r"\s+")

# Singkatan umum yang TIDAK boleh dianggap akhir kalimat
_ABBREVIATIONS = {
    "dr", "drs", "dra", "prof", "ir", "h", "hj", "kh", "st", "no", "jl", "jln",
    "rp", "tbk", "dkk", "dll", "dsb", "dst", "sdr", "bpk", "ibu", "yth", "kec",
    "kab", "kel", "prov", "pt", "cv", "mr", "mrs", "ms", "vs", "sh", "se",
    "mh", "mm", "mt", "phd", "kom", "pd", "sp", "spd", "hlm", "tgl", "a.n",
    "u.p", "s.h", "s.e", "s.pd", "s.kom", "m.si", "m.a", "lt", "kol", "letjen",
    "mayjen", "brigjen", "kombes", "ajun", "irjen", "komjen", "ipda", "aiptu",
    "pol", "akbp", "kompol", "aipda", "bripka", "brigpol", "jend", "marsekal",
}
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"”’]?\s+(?=[\"“]?[A-Z0-9])")


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_articles(filepath: Path | str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Memuat dataset berita dan melakukan pembersihan tingkat-dokumen.

    Args:
        filepath: path ke data.csv
        max_rows: batasi jumlah baris (None = semua)

    Returns:
        DataFrame kolom: id, source, title, url, date, content, summary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {filepath}\n"
            "Letakkan 'data.csv' di folder root project atau gunakan --data <path>."
        )

    header = pd.read_csv(filepath, nrows=0).columns
    usecols = [c for c in USECOLS if c in header]
    missing = {"content", "title"} - set(usecols)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {sorted(missing)}")

    df = pd.read_csv(filepath, usecols=usecols, nrows=max_rows)
    return preprocess_dataset(df)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan DataFrame artikel:
      - buang baris dengan content kosong
      - bersihkan boilerplate pada title/content/summary
      - buang artikel yang terlalu pendek (< 30 kata) dan duplikat konten
    """
    df = df.copy()
    for col in ("source", "url", "date", "summary"):
        if col not in df.columns:
            df[col] = ""
    if "id" not in df.columns:
        df["id"] = range(len(df))

    df = df.dropna(subset=["content", "title"])
    df["title"] = df["title"].astype(str).map(normalize_whitespace)
    df["content"] = df["content"].astype(str).map(clean_text)
    df["summary"] = df["summary"].fillna("").astype(str).map(normalize_whitespace)
    df["source"] = df["source"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)

    df = df[df["content"].str.split().str.len() >= 30]
    df = df.drop_duplicates(subset=["content"])
    df = df.drop_duplicates(subset=["title", "source"])
    df["id"] = df["id"].astype(int)
    for col in ("title", "content", "summary", "source", "url", "date"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Decode entitas HTML dan rapikan spasi."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).replace("\xa0", " ")
    return _WS_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """
    Membersihkan satu artikel berita dari noise hasil scraping.

    >>> clean_text('TEMPO.CO, Jakarta - Api padam.Warga lega. Pilihan Editor: X')
    'Api padam. Warga lega.'
    """
    text = normalize_whitespace(text)
    if not text:
        return ""

    for pattern in _DATELINE_RE:
        new = pattern.sub("", text, count=1)
        if new != text:
            text = new
            break

    for pattern in _NOISE_RE:
        text = pattern.sub(" ", text)

    text = _GLUED_SENT_RE.sub(r"\1 ", text)
    return _WS_RE.sub(" ", text).strip()


# -----------------------------------------------------------------------------
# Segmentasi kalimat & passage
# -----------------------------------------------------------------------------
def split_sentences(text: str, min_words: int = 3) -> List[str]:
    """
    Memecah teks menjadi kalimat dengan memperhatikan singkatan bahasa Indonesia
    (mis. "Rp.", "Dr.", "Kombes.", inisial "M.") agar tidak salah potong.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    pieces = _SENT_BOUNDARY_RE.split(text)
    sentences: List[str] = []
    buffer = ""
    for piece in pieces:
        buffer = f"{buffer} {piece}".strip() if buffer else piece.strip()
        last_word = buffer.rstrip("\"”’").rsplit(" ", 1)[-1].rstrip(".").lower()
        is_initial = len(last_word) == 1 and last_word.isalpha()
        if last_word in _ABBREVIATIONS or is_initial:
            continue  # gabungkan dengan potongan berikutnya
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)

    # Kalimat yang terlalu pendek digabung ke kalimat sebelumnya
    merged: List[str] = []
    for sent in sentences:
        if merged and len(sent.split()) < min_words:
            merged[-1] = f"{merged[-1]} {sent}"
        else:
            merged.append(sent)
    return merged


def make_passages(
    text: str,
    sentences_per_passage: int = 3,
    stride: int = 2,
    max_words: int = 120,
) -> List[str]:
    """
    Membentuk passage evidence dengan sliding window kalimat.

    Passage pendek (±3 kalimat) jauh lebih cocok untuk NLI dibanding artikel
    utuh: premis menjadi fokus dan tidak terpotong batas token model.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    stride = max(1, stride)
    passages: List[str] = []
    last_start = max(0, len(sentences) - sentences_per_passage)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)  # pastikan ekor artikel ikut tercakup
    for start in starts:
        passage = " ".join(sentences[start : start + sentences_per_passage])
        passages.append(truncate_text(passage, max_words=max_words))
    return passages


def truncate_text(text: str, max_words: int = 150) -> str:
    """Memotong teks menjadi maksimal `max_words` kata."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " ..."
    return text
