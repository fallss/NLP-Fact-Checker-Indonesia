# 📝 DAFTAR CONTOH INPUT KLAIM PENGUJIAN (FACT-CHECKER)

Dokumen ini menyediakan kumpulan contoh klaim uji coba berbahasa Indonesia yang dirancang khusus untuk menguji keandalan sistem **Indonesian News Fact-Checker v2.0**. Klaim-klaim ini diambil dan disesuaikan dari peristiwa nasional nyata pada korpus berita **Maret–April 2023** (`data.csv`).

Anda dapat menyalin (*copy*) salah satu klaim di bawah ini dan menempelkannya (*paste*) pada:
- **Web App Gradio**: Masukkan pada kotak teks input klaim di `http://127.0.0.1:7860`.
- **Terminal CLI**: `python -m fact_checker check "<klaim>"`
- **Jupyter Notebook**: Pada sel demo di `Fact_Checker_Full_Project.ipynb`.

---

## 🟢 1. Kategori Fakta Benar (Expected Verdict: `DIDUKUNG FAKTA` / `SUPPORTED`)
Klaim-klaim berikut berisi fakta yang **selaras dan terbukti secara eksplisit** di dalam artikel berita korpus.

### Klaim 1.1: Kebakaran Depo Pertamina Plumpang
> `Presiden Joko Widodo telah memerintahkan Wakil Presiden Ma'ruf Amin untuk meninjau langsung lokasi kebakaran depo Pertamina di Plumpang.`
- **Portal Terkait**: *Tempo, Detikcom* (4 Maret 2023).
- **Fakta dalam Berita**: Deputi Bidang Protokol, Pers, dan Media Sekretariat Presiden Bey Machmudin membenarkan arahan Presiden Jokowi kepada Wapres Ma'ruf Amin untuk meninjau penanganan korban Plumpang.

### Klaim 1.2: Pembatalan Tuan Rumah Piala Dunia U-20
> `FIFA secara resmi mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20 2023.`
- **Portal Terkait**: *Kompas, CNN Indonesia, Detikcom* (29 Maret 2023).
- **Fakta dalam Berita**: FIFA mengeluarkan pernyataan resmi membatalkan Indonesia sebagai tuan rumah Piala Dunia U-20 setelah pertemuan Presiden FIFA Gianni Infantino dengan Ketua Umum PSSI Erick Thohir di Doha.

### Klaim 1.3: Penetapan 1 Ramadan 1444 Hijriah
> `Pemerintah menetapkan 1 Ramadan 1444 Hijriah jatuh pada hari Kamis, 23 Maret 2023.`
- **Portal Terkait**: *Republika, Antara News* (22 Maret 2023).
- **Fakta dalam Berita**: Sidang Isbat Kementerian Agama RI yang dipimpin Menteri Agama Yaqut Cholil Qoumas menyepakati awal puasa Ramadan 1444 H jatuh pada Kamis, 23 Maret 2023.

### Klaim 1.4: Pertemuan Prabowo Subianto dan Surya Paloh
> `Ketua Umum Partai Gerindra Prabowo Subianto dan Ketua Umum Partai NasDem Surya Paloh sepakat saling menghormati keputusan politik masing-masing dalam Pemilu 2024.`
- **Portal Terkait**: *Detikcom, Tempo* (5 Maret 2023).
- **Fakta dalam Berita**: Pertemuan bilateral di Hambalang menghasilkan kesepakatan bahwa Gerindra dan NasDem menghormati jalan politik koalisi masing-masing demi stabilitas nasional.

### Klaim 1.5: Banding KPU atas Putusan PN Jakarta Pusat
> `KPU menyatakan akan mengajukan banding terhadap putusan PN Jakarta Pusat terkait penundaan tahapan Pemilu 2024.`
- **Portal Terkait**: *Kompas, Tempo* (2 Maret 2023).
- **Fakta dalam Berita**: Ketua KPU Hasyim Asy'ari menegaskan KPU mengajukan banding atas gugatan Partai Prima yang dikabulkan PN Jakpus.

---

## 🔴 2. Kategori Fakta Salah / Hoaks (Expected Verdict: `BERTENTANGAN / HOAKS` / `REFUTED`)
Klaim-klaim berikut berisi informasi yang **berkebalikan (*kontradiksi langsung*)** dengan fakta yang dilaporkan oleh media.

### Klaim 2.1: Pemutarbalikan Fakta Depo Plumpang
> `Presiden Joko Widodo melarang keras Wakil Presiden Ma'ruf Amin untuk mengunjungi lokasi kebakaran depo Pertamina di Plumpang.`
- **Ekspektasi Sistem**: Terdeteksi **BERTENTANGAN / HOAKS** (Confidence ~99-100%).
- **Alasan Logika**: Berita mencatat kata *"memerintahkan"*, sedangkan klaim memutarbalikkan menjadi *"melarang keras"*. Model NLI akan mengenali kontradiksi semantik ini dan SHAP akan menyoroti kata *"melarang"* vs *"memerintahkan"*.

### Klaim 2.2: Disinformasi Vonis Tragedi Kanjuruhan
> `Ketua Panpel Arema FC Abdul Haris divonis bebas dalam sidang putusan kasus Tragedi Kanjuruhan di Pengadilan Negeri Surabaya.`
- **Ekspektasi Sistem**: Terdeteksi **BERTENTANGAN / HOAKS**.
- **Alasan Logika**: Majelis Hakim PN Surabaya menjatuhkan vonis 1 tahun 6 bulan penjara kepada Abdul Haris, bukan vonis bebas.

### Klaim 2.3: Hoaks Penolakan Banding KPU
> `KPU pasrah dan menerima begitu saja putusan PN Jakarta Pusat untuk menunda seluruh tahapan Pemilu 2024 tanpa mengajukan upaya hukum banding.`
- **Ekspektasi Sistem**: Terdeteksi **BERTENTANGAN / HOAKS**.
- **Alasan Logika**: Berita resmi memberitakan KPU secara tegas mendaftarkan memori banding dan tetap menjalankan tahapan pemilu.

### Klaim 2.4: Manipulasi Korban Bencana Longsor
> `Bencana tanah longsor di Pulau Serasan Kabupaten Natuna tidak menimbulkan korban jiwa sama sekali.`
- **Ekspektasi Sistem**: Terdeteksi **BERTENTANGAN / HOAKS**.
- **Alasan Logika**: Berita BNPB dan Basarnas mencatat puluhan warga meninggal dunia dan belasan lainnya dinyatakan hilang.

### Klaim 2.5: Pembalikan Sikap Koalisi Politik
> `Prabowo Subianto dan Surya Paloh sepakat untuk saling bermusuhan secara terbuka dan memboikot pelaksanaan Pemilu 2024.`
- **Ekspektasi Sistem**: Terdeteksi **BERTENTANGAN / HOAKS**.
- **Alasan Logika**: Berita menyatakan mereka sepakat saling menghormati dan menjaga persahabatan, bertentangan dengan kata "memboikot" dan "bermusuhan".

---

## 🟡 3. Kategori Bukti Tidak Cukup / Netral (Expected Verdict: `BUKTI TIDAK CUKUP` / `NOT ENOUGH INFO`)
Klaim-klaim berikut membahas topik yang mirip, namun detail klaimnya **tidak diverifikasi atau tidak tercantum** di dalam berita korpus.

### Klaim 3.1: Klaim Angka / Nominal yang Dikarang
> `Pemerintah memberikan santunan uang tunai sebesar Rp 10 Miliar untuk setiap kepala keluarga korban kebakaran Plumpang.`
- **Ekspektasi Sistem**: Terdeteksi **BUKTI TIDAK CUKUP** (*Confidence moderate*).
- **Alasan Logika**: Di korpus memang banyak berita santunan dan bantuan sewa rumah untuk korban Plumpang, namun angka bombastis *"Rp 10 Miliar per KK"* tidak pernah ada di berita mana pun. Model NLI akan memberikan probabilitas Netral tertinggi.

### Klaim 3.2: Klaim Prediksi Masa Depan (Out of Context)
> `Timnas sepak bola Indonesia dipastikan menjuarai Piala Dunia pada tahun 2030 mendatang.`
- **Ekspektasi Sistem**: Terdeteksi **BUKTI TIDAK CUKUP**.
- **Alasan Logika**: Peristiwa masa depan belum terjadi dan tidak ada satu pun artikel berita fakta yang memvalidasi klaim tersebut.

### Klaim 3.3: Tokoh Pendamping Fiktif
> `Wakil Presiden Ma'ruf Amin meninjau lokasi kebakaran Plumpang didampingi oleh Gubernur Jawa Barat Ridwan Kamil.`
- **Ekspektasi Sistem**: Terdeteksi **BUKTI TIDAK CUKUP** / Ambang Ragu.
- **Alasan Logika**: Berita menyebutkan Ma'ruf Amin didampingi oleh Menteri BUMN Erick Thohir dan PJ Gubernur DKI Heru Budi, bukan Ridwan Kamil.

### Klaim 3.4: Peristiwa di Luar Rentang Waktu Korpus
> `Pemerintah Indonesia meresmikan pemindahan ibu kota negara secara penuh ke IKN Nusantara pada bulan Agustus 2024.`
- **Ekspektasi Sistem**: Terdeteksi **BUKTI TIDAK CUKUP**.
- **Alasan Logika**: Korpus berita dibatasi pada Maret–April 2023. Peristiwa Agustus 2024 berada di luar batas temporal dataset.

---

## 💡 Tips & Trik Pengujian untuk Pengguna:

1. **Uji Ketahanan Parafrase**:
   Coba ketik klaim dengan gaya bahasa santai atau susunan kata yang diacak, misalnya:
   *`"Gara-gara keputusan FIFA, Indonesia batal jadi tuan rumah piala dunia U20 tahun 2023."`*
   Lihat bagaimana model *Sentence-BERT* dan *mDeBERTa-v3* tetap mampu mengenali makna semantik kalimat tersebut!
2. **Perhatikan Highlight SHAP**:
   Pada klaim hoaks Plumpang (Klaim 2.1), buka bagian visualisasi SHAP di Web App. Anda akan melihat kata *"melarang"* disorot dengan warna kontras karena kata tersebut yang secara matematis mendorong AI memprediksi *CONTRADICTION*.
3. **Cek Tautan Sumber Asli**:
   Klik ikon panah kecil (`[↗]`) di pojok kanan atas setiap kartu bukti untuk melihat tautan rujukan artikel aslinya.
