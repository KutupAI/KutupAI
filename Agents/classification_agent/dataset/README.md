# Classification Agent -- Dataset Pipeline

Bu klasör, gerçek etiketlenmiş veri geldiğinde tek komutla bağlanacak
şekilde hazırlanmıştır. Kod yazmaya gerek yok -- sadece aşağıdaki adımları
sırayla çalıştır.

## 1) PDF + OCR JSON'ları eşleştir, etiketleme şablonu üret

```bash
python -m Agents.classification_agent.dataset.loader \
    --pdf-dir /path/to/pdfs \
    --ocr-dir /path/to/ocr_json \
    --output Agents/classification_agent/dataset/manifest_template.csv
```

`--ocr-dir` opsiyonel -- yoksa `ocr_json_path` boş kalır, classification_agent
görsel-only moda düşer.

## 2) `manifest_template.csv`'yi Excel/Sheets'te aç, `label` sütununu doldur

Geçerli kodlar (`Agents/classification_agent/taxonomy.py`):
dilekce, basvuru_belgesi, talep_yazisi, sikayet_basvurusu, itiraz_basvurusu,
bilgi_edinme_basvurusu, resmi_yazi, ust_yazi, izin_belgesi, onay_belgesi,
tutanak, form, beyan_beyanname, bildirim_tebligat, rapor,
karar_karar_yazisi, sozlesme_protokol, diger_belirsiz

Zor bir örnekse (§9), `hard_case_tags` sütununa `|` ile ayırarak kod ekle
(bkz. `evaluation/hard_cases.py` -- örn. `skewed_pdf|ocr_char_errors`).

Sentetik/üretilmiş bir örnekse `is_synthetic` sütununu `1` yap -- bu satırlar
otomatik olarak test setinin dışında tutulur (§6 kuralı).

## 3) Sınıf dağılım tablosunu üret

```bash
python -m Agents.classification_agent.dataset.distribution \
    --manifest Agents/classification_agent/dataset/manifest_template.csv \
    --output Agents/classification_agent/dataset/distribution_report.md
```

0 örnekli veya az örnekli sınıflar burada uyarı olarak çıkar -- §5/§6 gereği
bunlar gözden geçirilmeli.

## 4) Stratified train/val/test split

```bash
python -m Agents.classification_agent.dataset.splitter \
    --manifest Agents/classification_agent/dataset/manifest_template.csv \
    --output-dir Agents/classification_agent/dataset/splits
```

`splits/train.csv`, `splits/val.csv`, `splits/test.csv` üretilir.

## 5) Değerlendirme + rapor

`Agents/classification_agent/evaluation/` altındaki `metrics.py`,
`hard_cases.py`, `ablation.py`, `report.py` gerçek tahminlerle beslenmeye
hazır -- bkz. o klasörün docstring'leri.
