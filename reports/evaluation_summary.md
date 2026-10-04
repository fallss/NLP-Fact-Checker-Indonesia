# Hasil Evaluasi Fact-Checker: IndoBERT & Entailment Verification

## 1. NLI dengan evidence emas

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | detik/pasangan |
|---|---|---|---|---|---|
| `IndoBERT NLI / Multilingual Verifier` | 88.9% | 0.891 | 0.889 | **0.890** | 2.386 |
| `cross-encoder/nli-MiniLM2-L6-H768` (Baseline) | 42.2% | 0.278 | 0.422 | **0.326** | 0.095 |

<details><summary>Laporan per kelas — IndoBERT & Multilingual Verifier</summary>

```
               precision    recall  f1-score   support

   ENTAILMENT      0.933     0.933     0.933        15
      NEUTRAL      0.929     0.867     0.897        15
CONTRADICTION      0.812     0.867     0.839        15

     accuracy                          0.889        45
    macro avg      0.891     0.889     0.890        45
 weighted avg      0.891     0.889     0.890        45
```
</details>

<details><summary>Laporan per kelas — baseline</summary>

```
               precision    recall  f1-score   support

   ENTAILMENT      0.433     0.867     0.578        15
      NEUTRAL      0.000     0.000     0.000        15
CONTRADICTION      0.400     0.400     0.400        15

     accuracy                          0.422        45
    macro avg      0.278     0.422     0.326        45
 weighted avg      0.278     0.422     0.326        45
```
</details>

## 2. Retrieval artikel emas

| Metode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 17.8% | 40.0% | 44.4% | 53.3% | 0.293 |
| dense | 22.2% | 35.6% | 44.4% | 53.3% | 0.311 |
| hybrid | 28.9% | 51.1% | 53.3% | 60.0% | 0.398 |

## 3. End-to-end (klaim → verdict)

Waktu rata-rata: 14.43 detik/klaim · evidence emas terambil: 44.4%

| Strategi agregasi | Accuracy | Macro-F1 |
|---|---|---|
| max | 66.7% | 0.655 |
| weighted | 66.7% | 0.660 |
| mean | 68.8% | 0.686 |

```
                 precision    recall  f1-score   support

      SUPPORTED      0.688     0.733     0.710        15
        REFUTED      0.583     0.933     0.718        15
NOT_ENOUGH_INFO      0.875     0.389     0.538        18

       accuracy                          0.667        48
      macro avg      0.715     0.685     0.655        48
   weighted avg      0.725     0.667     0.648        48
```

### Kesalahan prediksi (strategi max)

| # | Klaim | Emas | Prediksi |
|---|---|---|---|
| 1 | Presiden Jokowi memerintahkan Wakil Presiden Ma'ruf Amin meninjau lokasi kebakaran Depo Pertamina Plumpang. | SUPPORTED | REFUTED |
| 3 | Wakil Presiden Ma'ruf Amin meninjau lokasi kebakaran Plumpang bersama Gubernur Jawa Barat Ridwan Kamil. | NOT_ENOUGH_INFO | REFUTED |
| 6 | Kebakaran Depo Plumpang dipastikan disebabkan oleh sambaran petir. | NOT_ENOUGH_INFO | REFUTED |
| 9 | Pengadilan Tinggi DKI Jakarta telah mengabulkan banding yang diajukan KPU. | NOT_ENOUGH_INFO | SUPPORTED |
| 12 | Abdul Haris langsung mengajukan banding ke Pengadilan Tinggi Jawa Timur setelah vonis dibacakan. | NOT_ENOUGH_INFO | REFUTED |
| 16 | Warga diminta menjauhi radius 7 km dari puncak Merapi setelah awan panas guguran pada 11 Maret 2023. | SUPPORTED | NOT_ENOUGH_INFO |
| 21 | Rafael Alun Trisambodo telah divonis 14 tahun penjara oleh Pengadilan Tipikor. | NOT_ENOUGH_INFO | SUPPORTED |
| 23 | Kejati DKI Jakarta menyetujui penyelesaian kasus Mario Dandy melalui restorative justice. | REFUTED | SUPPORTED |
| 33 | Pemerintah memberlakukan larangan mudik bagi aparatur sipil negara pada Lebaran 2023. | NOT_ENOUGH_INFO | SUPPORTED |
| 36 | Pendaftaran mudik gratis Kemenhub hanya dapat dilakukan melalui kantor pos. | NOT_ENOUGH_INFO | REFUTED |
| 39 | FIFA menjatuhkan denda sebesar 10 juta dolar AS kepada PSSI. | NOT_ENOUGH_INFO | REFUTED |
| 40 | BMKG melaporkan gempa bermagnitudo 6,3 mengguncang Keerom, Papua. | SUPPORTED | REFUTED |
| 43 | Polisi menangkap 18 pelajar yang hendak tawuran di Kalideres, Jakarta Barat. | SUPPORTED | REFUTED |
| 45 | Para pelajar yang ditangkap di Kalideres berasal dari tiga sekolah kejuruan yang berbeda. | NOT_ENOUGH_INFO | SUPPORTED |
| 46 | Timnas sepak bola Indonesia dipastikan menjuarai Piala Dunia 2030. | NOT_ENOUGH_INFO | REFUTED |
| 47 | Peneliti Indonesia menemukan spesies dinosaurus baru di Pulau Sumba. | NOT_ENOUGH_INFO | REFUTED |
