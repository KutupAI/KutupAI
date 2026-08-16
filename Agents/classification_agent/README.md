# Classification Agent

Belge görüntüsü + OCR metni + layout bilgisini birlikte kullanarak belgeyi
18 sınıftan birine ayıran Agent. Teknik görev dosyasındaki gereksinimlere
göre inşa edilmiştir (`Documentation/agent_catalog.md` ve `architecture.md`
ile uyumlu).

Diğer Agent'lar gibi: `BaseAgent`'tan türer, sadece `graph_state`'i günceller
(`classification_result`), Storage'a hiç yazmaz.

---

## 1) Kurulum

```bash
pip install -r Agents/classification_agent/requirements.txt
```

## 2) Qwen VLM modelini indir ve çalıştır

```bash
pip install -U "huggingface_hub[cli]"

hf download unsloth/Qwen2.5-VL-3B-Instruct-GGUF \
  --local-dir Inference/models \
  --include "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" "mmproj-F16.gguf"
```

Dosyaları beklenen adlara çevir:

```bash
mv Inference/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf Inference/models/qwen-vl.gguf
mv Inference/models/mmproj-F16.gguf Inference/models/qwen-vl-mmproj.gguf
```

`llama.cpp` sunucusunu indir (https://github.com/ggml-org/llama.cpp/releases,
işletim sistemine uygun `bin-*.zip`) ve **8092** portunda başlat:

```bash
llama-server \
  -m Inference/models/qwen-vl.gguf \
  --mmproj Inference/models/qwen-vl-mmproj.gguf \
  --host 0.0.0.0 --port 8092 --ctx-size 8192 \
  --image-min-tokens 1024
```

Doğrulama:

```bash
curl http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"merhaba"}],"max_tokens":20}'
```

## 3) Etiketli veri hazırlığı

Gerçek PDF + OCR JSON'lar geldiğinde -- ayrıntılı adımlar için
[`dataset/README.md`](dataset/README.md):

```bash
python -m Agents.classification_agent.dataset.loader --pdf-dir <PDF_KLASORU> --ocr-dir <OCR_JSON_KLASORU>
# manifest_template.csv'yi doldur (label sütunu), sonra:
python -m Agents.classification_agent.dataset.distribution --manifest Agents/classification_agent/dataset/manifest_template.csv
python -m Agents.classification_agent.dataset.splitter --manifest Agents/classification_agent/dataset/manifest_template.csv
```

## 4) Agent'ı Python'dan çalıştırma (manuel test)

```python
from Agents.classification_agent.agent import ClassificationAgent
from Agents.classification_agent.config import ClassificationConfig

agent = ClassificationAgent(config=ClassificationConfig())
state = {
    "document_id": "DOC-001",
    "ocr_result": {
        "full_text": "...",
        "pages": [{"text_items": [{"text": "x", "confidence": 0.9}]}],
    },
}
result = agent.run(state)
print(result["classification_result"])
```

Not: `ClassificationAgent`'ı import etmek `Agents/__init__.py` üzerinden
`ocr_agent`'ı da tetikler -- `Agents/ocr_agent/requirements.txt`
(OpenCV dahil) kurulu olmalı.

## 5) Değerlendirme (gerçek veri geldikten sonra)

`evaluation/metrics.py`, `evaluation/hard_cases.py`, `evaluation/ablation.py`,
`evaluation/report.py` -- her modülün docstring'inde kullanım şekli var.
Nihai teslim raporu `evaluation/report.py::build_report()` ile üretilir.

---

## Dosya haritası

```
Agents/classification_agent/
├── agent.py           # ana Agent (BaseAgent alt sınıfı)
├── config.py           # threshold'lar (needs_review, escalation), Qwen ayarları
├── prompts.py           # Qwen VLM prompt şablonu
├── tools.py            # Optimization + Qwen VLM çağrı katmanı
├── models.py           # çıktı JSON şeması (dataclasses)
├── taxonomy.py           # 18 sınıfın tek doğru kaynağı
├── exceptions.py         # Agent'a özel hatalar
├── dataset/            # manifest, dağılım tablosu, train/val/test split
└── evaluation/           # metrics, hard-case breakdown, ablation, rapor
```

---

## Bu Agent ne yapıyor

**Classification Agent**, bir belgenin (dilekçe, başvuru, şikâyet, rapor vb.)
OCR metnini + görüntüsünü + layout bilgisini birlikte değerlendirip, 18
sınıftan hangisine ait olduğuna karar veren bağımsız bir Agent'tır.

Akış: `ocr_agent`'ın ürettiği sonuç (`graph_state["ocr_result"]`) girdi
olarak alınır → önce Optimization katmanındaki hızlı ONNX sınıflandırıcı
denenir (varsa) → yetersiz/yoksa **Qwen2.5-VL** modeline görüntü + metin +
layout birlikte gönderilir → model kesin bir JSON (`document_type`,
`confidence`, `alternatives`, `status`) döndürür → sonuç
`graph_state["classification_result"]`'a yazılır. Agent hiçbir zaman
Storage'a doğrudan yazmaz; sadece state'i günceller (projedeki tüm
Agent'ların ortak kuralı).

Qwen çalışırken 8092 portunda ayrı bir `llama-server` süreci olarak durur;
projedeki diğer tüm Agent'ların kullandığı Gemma3'e hiç dokunulmadı —
`model_registry.json`'a ikinci, bağımsız bir model girdisi olarak eklendi.

---

## Eklediğimiz her dosya ve işlevi

### `agent.py`
Agent'ın kalbi. `BaseAgent`'tan türeyen `ClassificationAgent` sınıfını
içerir. `run(state)` metodu şunları yapar: state'ten OCR metnini/görüntüyü/
layout'u çeker (`_extract_inputs`), önce hızlı sınıflandırıcıyı dener
(`run_fast_classifier`), yeterince emin değilse Qwen'e geçer
(`run_qwen_classification`), sonucu `needs_review` eşiğine göre
`success`/`needs_review` olarak işaretler ve `ClassificationResult`
nesnesini `state["classification_result"]`'a yazar. Girdi hiç yoksa
(ne metin ne görüntü) çökmeden `status: failed` döner.

### `config.py`
Tüm ayarlanabilir sayıları tek yerde toplar: `needs_review_threshold`
(şu an 0.60, **gerçek validation seti ile yeniden ayarlanmalı**),
`fast_classifier_escalation_threshold`, `top_k_alternatives`, Qwen sunucu
adresi (`http://localhost:8092/...`), zaman aşımı, görüntü gönderilsin mi
vb. Ortam değişkenlerinden okunabilir (`ClassificationConfig.from_env()`)
-- yani sunucu adresi/eşikler kod değiştirmeden `.env` ile ayarlanabilir.

### `taxonomy.py`
18 sınıfın **tek doğru kaynağı**. Her sınıf: makine-okunur kod
(`dilekce`, `talep_yazisi`, ...), Türkçe adı, sırası. Diğer tüm dosyalar
(prompt, evaluation, dataset) sınıf listesini buradan okur -- sınıflar
değişirse (§5 gereği, gerçek veriyle desteklenmeyenler kaldırılabilir)
tek dosya güncellenir, geri kalan kod otomatik uyum sağlar.

### `models.py`
Çıktı JSON'unun şeklini tanımlayan `dataclass`'lar
(`ClassificationResult`, `ClassificationAlternative`). `to_dict()` metodu,
teknik görev dosyasındaki §7 örneğiyle birebir aynı alan sırasını üretir,
artı iz sürme için ek alanlar (`source`: hangi katman karar verdi,
`ocr_confidence`, `processing_ms`).

### `prompts.py`
Qwen'e gönderilen sistem promptu + kullanıcı promptu şablonu. 18 sınıfı
`taxonomy.py`'den okuyup listeler, modele "sadece bu listeden seç, JSON
dışında hiçbir şey yazma" kuralını verir, OCR güveni düşükse modeli
görsel ipuçlarına daha çok ağırlık vermesi için uyarır. Layout bilgisini
(varsa) okunabilir bir metne çeviren `build_layout_summary()` da burada.

### `tools.py`
Agent ile dış sistemler (Optimization, Qwen) arasındaki köprü.
`run_fast_classifier()` Optimization katmanını çağırır. 
`run_qwen_classification()` Qwen'e isteği gönderir, dönen metinden JSON'u
ayıklar (`_parse_json_response`) ve **taksonomi dışı bir sınıf gelirse
onu asla kabul etmez** -- otomatik olarak `diger_belirsiz`'e düşürür.
Bu, modelin sınıf uydurmasına karşı tek güvenlik katmanı.

### `exceptions.py`
Agent'a özel hata sınıfları (`MissingInputError`, `QwenVLMError`,
`InvalidClassificationOutputError` vb.) -- `agent.py`'nin hataları
ayırt edip her durumda düzgün bir `status: failed` JSON'u dönebilmesi için.

### `requirements.txt`
Bu Agent'ın ek Python bağımlılıkları (`requests`, opsiyonel `scikit-learn`).

### `__init__.py`
Paketi `Agents/__init__.py` üzerinden Supervisor'ün keşfedebilmesi için
gerekli import/registration.

### `Inference/client/qwen_vl_client.py` *(classification_agent dışında ama onun için eklendi)*
`llama_client.py` (Gemma3 için kullanılan) ile aynı tarzda, ama görüntü
gönderebilen ayrı bir HTTP istemcisi. Neden ayrı: Gemma3 client'ı
metin-only; Qwen görüntü + metin birlikte istiyor. Diğer Agent'lar hâlâ
`llama_client.py`'yi kullanıyor, hiçbiri etkilenmedi.

### `dataset/schema.py`
Etiketleme tablosunun (manifest) satır şekli: `document_id`, `pdf_path`,
`ocr_json_path`, `label`, `is_synthetic`, `hard_case_tags`, `split`.
Excel/Sheets'te açılabilir düz bir CSV'ye karşılık gelir.

### `dataset/loader.py`
Gerçek PDF klasörünü (+ opsiyonel OCR JSON klasörünü) tarar, eşleştirir,
boş `label` sütunlu bir CSV şablonu üretir (`manifest_template.csv`) --
ekibin sadece etiket sütununu doldurması yeterli. Doldurulmuş manifestoyu
geri okuyup taksonomi dışı etiketleri hata olarak yakalar.

### `dataset/distribution.py`
18 sınıf için örnek sayısı tablosunu üretir (§6 deliverable'ı), 0 veya az
örnekli sınıfları otomatik işaretler.

### `dataset/splitter.py`
Stratified train/val/test bölme. Sentetik (`is_synthetic=True`) belgeleri
otomatik olarak test setinden hariç tutar (§6 kuralı).

### `evaluation/metrics.py`
Accuracy, Macro-F1, Weighted-F1, sınıf başına precision/recall/f1,
confusion matrix, gecikme (latency) istatistikleri -- hepsi §8'in istediği
metrikler, sklearn'e ihtiyaç duymadan.

### `evaluation/hard_cases.py`
§9'daki 10 zor senaryonun (yamuk PDF, düşük çözünürlük, el yazısı, vb.)
birebir listesi + bu senaryolara göre etiketlenmiş belgeler üzerinde
ayrı accuracy hesaplayan `breakdown_by_hard_case()`.

### `evaluation/ablation.py`
§10'daki zorunlu karşılaştırmaların (OCR-only / image-only / OCR+image /
+layout) çalıştırma iskeleti. Model boyutu ve class-balancing
karşılaştırmaları için TODO olarak işaretli (henüz o artefaktlar yok).

### `evaluation/report.py`
Yukarıdaki her şeyi (dağılım, split, metrikler, hard-case sonuçları,
ablation) §11'in teslimat listesiyle birebir eşleşen tek bir Markdown
rapora birleştirir. Eksik veri olan bölüm "PENDING" olarak işaretlenir,
asla uydurulmaz.