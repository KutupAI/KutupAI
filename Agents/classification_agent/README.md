# Classification Agent

## Güncel proje ayarı

Çalışan entegrasyonda sınıflandırma, OCR metni ve mevcut layout sinyallerini EVREN `llm-fast` modeline gönderir. `CLASSIFICATION_INFERENCE_BACKEND=evren` ve `CLASSIFICATION_EVREN_MODEL=llm-fast` varsayılandır. Bu seçim yalnız model istemcisini değiştirir; `ClassificationAgent.run(state)` sözleşmesi ve çıktısı aynıdır.

Yerel Gemma / ONNX ile ilgili aşağıdaki bölümler, alternatif veya geçmiş geliştirme akışları için korunmuştur. Yerel moda dönmek için `CLASSIFICATION_INFERENCE_BACKEND=local` kullanılır.

Belge görüntüsü + OCR metni + layout bilgisini birlikte kullanarak belgeyi
18 sınıftan birine ayıran Agent. `BaseAgent`'tan türer, sadece
`graph_state`'i günceller (`classification_result` + `classification`),
Storage'a hiç yazmaz.

> **Model notu:** Başlangıçta Qwen2.5-VL kullanıyordu, artık **Gemma 3**
> (yerel llama.cpp/llama-server) kullanıyor. Kod mantığı değişmedi, sadece
> isimlendirme genellendi (`Qwen*` → `VLM*`). Eski isimler hâlâ çalışıyor
> (geriye dönük uyumluluk) -- ayrıntı: bölüm "Dosya haritası ve işlevleri".

---

## ÇALIŞTIRMA SIRASI (baştan sona, sırayla)

### 1 · Kur
```bash
pip install -r Agents/classification_agent/requirements.txt
```

### 2 · Model dosyalarını indir (bir kere)
```bash
pip install -U "huggingface_hub[cli]"

hf download bartowski/google_gemma-3-27b-it-GGUF \
  --local-dir Inference/models \
  --include "google_gemma-3-27b-it-Q4_K_M.gguf" "mmproj-google_gemma-3-27b-it-f32.gguf"

mv "Inference/models/google_gemma-3-27b-it-Q4_K_M.gguf" Inference/models/gemma-3.gguf
mv "Inference/models/mmproj-google_gemma-3-27b-it-f32.gguf" Inference/models/gemma-3-mmproj.gguf
```
*(Daha küçük/hızlı istersen 27B yerine 4B/12B indir -- dosya adları
değişir ama `gemma-3.gguf` / `gemma-3-mmproj.gguf` isimlerine çevirmen ve
adım 3'teki `--alias`'ı aynı bırakman yeterli.)*

### 3 · Sunucuyu başlat (ayrı terminal, açık bırak)
```bash
llama-server \
  -m Inference/models/gemma-3.gguf \
  --mmproj Inference/models/gemma-3-mmproj.gguf \
  --host 0.0.0.0 --port 8092 --ctx-size 8192 \
  --image-min-tokens 1024 \
  --alias gemma-3-27b-it
```
⚠️ **`--mmproj` olmadan görsel sessizce yok sayılır, hata vermez.**
`extraction_agent` de aynı sunucuyu (port 8092) paylaşır -- ikinci bir
sunucu başlatmana gerek yok.

Doğrulama (istersen):
```bash
curl http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"merhaba"}],"max_tokens":20}'
```

### 4 · Agent'ı çalıştır (yeni terminal, proje kökünden)
```bash
# Offline demo (unified envelope, mocked model)
python Tests/Agents/manual_test_classification.py

# Live Inference Gemma (Inference/llama_server on :8080)
python Tests/Agents/manual_test_classification.py --live

# Contract unit tests
pytest Tests/Agents/test_classification_agent.py -q
```
Fast classifier'ı kapatmak istersen:
```bash
# PowerShell
$env:CLASSIFICATION_USE_FAST_CLASSIFIER = "false"; python Tests/Agents/manual_test_classification.py --live
# bash
CLASSIFICATION_USE_FAST_CLASSIFIER=false python Tests/Agents/manual_test_classification.py --live
```

### 5 · (Opsiyonel) Etiketli veri hazırlığı
```bash
python -m Agents.classification_agent.dataset.loader --pdf-dir <PDF_KLASORU> --ocr-dir <OCR_JSON_KLASORU>
python -m Agents.classification_agent.dataset.distribution --manifest Agents/classification_agent/dataset/manifest_template.csv
python -m Agents.classification_agent.dataset.splitter --manifest Agents/classification_agent/dataset/manifest_template.csv
```
Ayrıntı: [`dataset/README.md`](dataset/README.md).

### 6 · (Opsiyonel) Python'dan manuel test
```python
from Agents.classification_agent.agent import ClassificationAgent
from Agents.classification_agent.config import ClassificationConfig

agent = ClassificationAgent(config=ClassificationConfig.from_env())
state = {"document_id": "DOC-001", "ocr_result": {"full_text": "..."}}
result = agent.run(state)
print(result["classification_result"])   # ayrıntılı
print(result["classification"])          # validation_agent'ın okuduğu kısa format
```
*(`ClassificationAgent` import etmek `ocr_agent`'ı da tetikler --
`Agents/ocr_agent/requirements.txt` kurulu olmalı.)*

### 7 · (Opsiyonel) Değerlendirme -- gerçek veri geldikten sonra
`evaluation/report.py::build_report()` -- her modülün kullanımı kendi
docstring'inde.

---

## Sorun mu çıktı? (Troubleshooting)

| Hata | Sebep | Çözüm |
|---|---|---|
| `ImportError: cannot import name 'clean_for_fast_classifier'` | `Optimization/services/preprocessing.py` boştu | Eksik fonksiyon eklendi -- düzeltildiyse bu satır silinebilir |
| `ModuleNotFoundError: ...vlm_client` | `Inference/client/vlm_client.py` eksik/yanlış yolda | Dosyanın tam bu isim+yolda olduğunu doğrula |
| VLM görsele tepki vermiyor, sadece metin | `--mmproj` olmadan başlatılmış | Sunucuyu adım 3'teki tam komutla yeniden başlat |

---

## Dosya haritası ve işlevleri

```
Agents/classification_agent/
├── agent.py          # ana Agent sınıfı
├── config.py          # threshold'lar, VLM ayarları
├── prompts.py          # VLM prompt şablonu
├── tools.py             # Optimization + VLM çağrı katmanı
├── models.py              # çıktı JSON şeması
├── taxonomy.py             # 18 sınıfın tek doğru kaynağı
├── exceptions.py            # Agent'a özel hatalar
├── dataset/                  # manifest, split
└── evaluation/                 # metrics, ablation, rapor
```

### `agent.py`
Agent'ın kalbi, kontrol akışının tamamı burada. `run(state)` metodu
sırasıyla: (1) state'ten OCR metnini, varsa belge görselini ve layout
bilgisini çeker (`_extract_inputs`); (2) önce Optimization katmanındaki
hızlı ONNX sınıflandırıcıyı dener (`run_fast_classifier`) -- bu ucuz ve
hızlı bir ön-filtre; (3) sonucu yoksa veya yeterince emin değilse
(eşiğin altındaysa) Gemma 3'e geçer (`run_vlm_classification`); (4) dönen
güven skorunu `needs_review_threshold` ile karşılaştırıp sonucu
`success` veya `needs_review` olarak işaretler -- düşük güvende bile en
iyi tahmini atmaz, sadece "gözden geçir" bayrağı ekler; (5) sonucu iki
yere yazar: `state["classification_result"]` (tüm ayrıntılarla) ve
`state["classification"]` (validation_agent'ın okuduğu kısa/özet format).
Girdi hiç yoksa (ne metin ne görsel) hata fırlatmadan `status: failed`
JSON'u döner -- pipeline'ın geri kalanı çökmeden devam edebilir.

### `config.py`
Tüm ayarlanabilir değerlerin tek toplandığı yer -- kod içinde hiçbir
"sihirli sayı" (magic number) yok, hepsi buradan okunuyor. En
önemlileri: `needs_review_threshold` (şu an 0.60 -- bu **tahmini** bir
değer, gerçek etiketli veri/validation seti geldiğinde yeniden
ayarlanmalı), `fast_classifier_escalation_threshold` (hızlı
sınıflandırıcının ne kadar emin olması gerektiği), `vlm_base_url`
(varsayılan `http://localhost:8092/...`, Gemma 3 sunucusunun adresi),
`vlm_model_name` (varsayılan `gemma-3-27b-it`, `llama-server`'daki
`--alias` ile eşleşmeli), zaman aşımı süresi, görsel gönderilsin mi vb.
Hepsi `.env` dosyasından okunabilir (`ClassificationConfig.from_env()`
ile) -- yani sunucu adresini veya eşikleri değiştirmek için kod
değiştirmene gerek yok. Eski `QWEN_VLM_*` isimli ortam değişkenleri de
hâlâ okunuyor (yeni `VLM_*` öncelikli, o yoksa eskiye bakılır) -- yani
eski bir `.env` dosyası güncellenmeden de çalışmaya devam eder.

### `taxonomy.py`
Projedeki 18 belge sınıfının **tek doğru kaynağı** (single source of
truth). Her sınıf üç şeyle tanımlı: makine-okunur kod (`dilekce`,
`talep_yazisi`, `form`, ...), Türkçe görünen adı, ve listedeki sırası.
Prompt, evaluation ve dataset modüllerinin hepsi sınıf listesini
doğrudan buradan okur -- yani sınıflardan biri kaldırılır veya eklenirse
(örneğin gerçek veriyle desteklenmeyen bir sınıf silinirse), tek bu
dosyayı güncellemen yeterli, geri kalan tüm kod otomatik olarak uyum
sağlar.

### `models.py`
Agent'ın çıktısının şeklini tanımlayan `dataclass`'lar:
`ClassificationResult` (ana sonuç -- `document_type`, `confidence`,
`alternatives`, `status`, `source` vb.) ve `ClassificationAlternative`
(en olası ikinci/üçüncü tahminler). `to_dict()` metodu bu nesneleri
JSON'a çevirir. `source` alanı hangi katmanın karar verdiğini gösterir:
artık `"optimization_fast"` veya `"vlm"` değerlerini alıyor (eskiden
`"qwen_vlm"` idi).

### `prompts.py`
Gemma 3'e gönderilen sistem promptu ve kullanıcı promptu şablonları
burada tanımlı. Model-agnostik yazılmış -- yani prompt metninin hiçbir
yerinde model adı geçmiyor, bu yüzden Qwen'den Gemma'ya geçişte bu
dosyada tek bir satır bile değişmedi. 18 sınıfı `taxonomy.py`'den
otomatik okuyup listeler, modele "sadece bu 18 sınıftan birini seç, JSON
dışında hiçbir açıklama/metin yazma" kuralını dayatır. OCR güveni
düşükse modeli görsel ipuçlarına daha çok ağırlık vermesi konusunda
uyarır. Layout bilgisini (varsa) okunabilir bir metne çeviren
`build_layout_summary()` fonksiyonu da burada.

### `tools.py`
Agent'ı dış sistemlere (Optimization katmanı, VLM sunucusu) bağlayan
köprü katmanı. `run_fast_classifier()` Optimization'daki ONNX modelini
çağırır (henüz eğitilmiş model yoksa `None` döner, hata vermez).
`run_vlm_classification()` (eski adı `run_qwen_classification` -- bu
isim hâlâ import edilebilir bir alias olarak duruyor, çünkü
`evaluation/ablation.py` hâlâ bu eski adı kullanıyor) Gemma 3'e isteği
`Inference/client/vlm_client.py` üzerinden gönderir, dönen ham metinden
JSON'u güvenli şekilde ayıklar (`_parse_json_response`), ve **en kritik
güvenlik kontrolünü** burada yapar: model taksonomi dışı bir sınıf ismi
uydurursa (örneğin listede olmayan bir kelime), bu asla kabul edilmez --
otomatik olarak `diger_belirsiz` sınıfına düşürülür. Bu, modelin sınıf
uydurmasına karşı tek koruma katmanı.

### `exceptions.py`
Agent'a özel hata sınıfları: `MissingInputError` (ne metin ne görsel
varsa), `VLMError` (eski adı `QwenVLMError`, alias olarak hâlâ
çalışıyor -- sunucu bağlantı hatası, timeout vb. için), ve
`InvalidClassificationOutputError` (model geçersiz/bozuk JSON döndüyse).
`agent.py` bu hataları yakalayıp her durumda düzgün bir `status: failed`
JSON'u döndürebilmek için kullanır -- yani hiçbir hata pipeline'ı
tamamen çökertmez.

### `requirements.txt`
Bu Agent'a özel ek Python bağımlılıkları: `requests` (HTTP istekleri
için), opsiyonel `scikit-learn` (ileride metrik hesaplamaları için).

### `__init__.py`
Paketi dışarı açar ve `Agents/__init__.py` üzerinden Supervisor'ün bu
Agent'ı otomatik keşfedip kaydedebilmesi (`@register` decorator'ı ile)
için gereken import zincirini kurar.

### `Inference/client/vlm_client.py` *(bu klasörün dışında ama bu Agent için var)*
Eskiden `qwen_vl_client.py` idi. `llama_client.py` (metin-only Gemma3
çağrıları için kullanılan istemci) ile aynı tarzda yazılmış, ama ek
olarak görsel de (base64 data URL şeklinde) gönderebilen ayrı bir HTTP
istemcisi. Önemli detay: bu dosyanın kodu **hiçbir zaman** Qwen'e özel
değildi -- düz, standart OpenAI-uyumlu chat-completions protokolü
kullanıyor. Bu yüzden Gemma 3'e geçişte tek değişen şey isimlendirme
oldu (`QwenVLClient` → `VLMClient`, `QwenVLRequest` → `VLMRequest`);
eski isimler dosyanın altında alias olarak duruyor, henüz güncellenmemiş
başka bir çağıran varsa kırılmasın diye. `extraction_agent`'ın görsel
çıkarım adımı da aynı istemci ailesini ve aynı 8092 portundaki sunucuyu
kullanıyor.

### `dataset/schema.py`
Etiketleme tablosunun (manifest) her satırının şeklini tanımlar:
`document_id`, `pdf_path`, `ocr_json_path`, `label`, `is_synthetic`
(yapay zekayla üretilmiş belge mi), `hard_case_tags`, `split` (train/val/
test). Excel veya Google Sheets'te doğrudan açılabilen düz bir CSV'ye
karşılık gelir.

### `dataset/loader.py`
Gerçek PDF klasörünü (+ varsa OCR JSON klasörünü) tarar, dosyaları
eşleştirir, ve `label` sütunu boş bırakılmış bir `manifest_template.csv`
şablonu üretir -- ekibin yapması gereken tek şey bu sütunu doldurmak.
Doldurulmuş manifestoyu geri okurken, taksonomi dışında bir etiket
girilmişse bunu hata olarak yakalar (yanlış yazılmış bir sınıf ismi
sessizce geçmesin diye).

### `dataset/distribution.py`
18 sınıfın her birinde kaç örnek olduğunu gösteren bir tablo üretir, 0
veya çok az örnekli sınıfları otomatik olarak işaretler -- dengesiz veri
setini erken fark etmek için.

### `dataset/splitter.py`
Train/validation/test bölmesini sınıf dağılımını koruyarak (stratified)
yapar. Yapay zekayla üretilmiş (`is_synthetic=True`) belgeleri otomatik
olarak test setinin dışında tutar -- gerçek dünya performansının yapay
verilerle şişirilmemesi için.

### `evaluation/metrics.py`
Accuracy, Macro-F1, Weighted-F1, sınıf başına precision/recall/f1,
confusion matrix (karışıklık matrisi), ve gecikme (latency)
istatistiklerini hesaplar -- hiçbir harici kütüphaneye (sklearn'e)
ihtiyaç duymadan, sıfırdan yazılmış.

### `evaluation/hard_cases.py`
Zor sınıflandırma senaryolarının (yamuk çekilmiş PDF, düşük çözünürlük,
el yazısı, karanlık/kalitesiz tarama vb.) sabit listesini tutar, ve bu
senaryolara göre etiketlenmiş belgeler üzerinde ayrı ayrı accuracy
hesaplayan `breakdown_by_hard_case()` fonksiyonunu içerir -- modelin
hangi zorluk türünde daha çok hata yaptığını görmek için.

### `evaluation/ablation.py`
Modelin farklı girdi kombinasyonlarıyla (sadece OCR metni / sadece
görsel / OCR+görsel / +layout bilgisi) ne kadar başarılı olduğunu
karşılaştıran test iskeleti -- hangi bilginin gerçekten fark yarattığını
ölçmek için. Hâlâ eski `run_qwen_classification` adını import ediyor,
ama `tools.py`'deki alias sayesinde sorunsuz çalışmaya devam ediyor.

### `evaluation/report.py`
Yukarıdaki tüm sonuçları (sınıf dağılımı, train/val/test bölmesi,
metrikler, zor-senaryo sonuçları, ablation karşılaştırmaları) tek bir
okunabilir Markdown raporunda birleştirir. Henüz veri olmayan bölümler
"PENDING" olarak işaretlenir -- hiçbir sonuç asla uydurulmaz.
