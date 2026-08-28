# Information Extraction Agent (Bilgi Cikarma Ajani)

## Güncel proje ayarı

Çalışan entegrasyonda metinden alan çıkarma EVREN `llm-fast` ile yapılır. Ayarlar `.env` içindeki `EXTRACTION_INFERENCE_BACKEND=evren` ve `EXTRACTION_LLM_MODEL=llm-fast` değişkenleridir. Agent yine aynı `state["extraction"]` sözleşmesini üretir; yalnızca çağrının gittiği model sağlayıcısı değişmiştir.

Bu dosyadaki yerel Gemma/8092 adımları geçmiş veya alternatif yerel çalışma içindir ve silinmemiştir. Yerel moda dönmek için backend değeri `local` yapılır.

Bir belgenin OCR metninden (ve varsa gorselinden) yapilandirilmis bilgi
cikaran Agent -- tarih, evrak numarasi, telefon, email gibi kesin
alanlari kural-tabanli (regex) yakalar; kisi adi, kurum, talep turu,
konu gibi anlamsal alanlari LLM (Gemma 3) ile cikarir; mumkun oldugunda
bunlari metinde birebir dogrulanmis (grounded) hale getirir (LangExtract).

> **Model degisikligi notu:** Baslangicta Qwen2.5 (LLM) + Qwen2.5-VL
> (Vision), Together AI (bulut) uzerinden calisiyordu. Iki degisiklik
> yapildi: (1) **Qwen -> Gemma 3** -- proje genelinde Gemma3 zaten
> varsayilan model oldugu icin; (2) **bulut -> yerel `llama-server`** --
> Turkce resmi belge metinlerinin disariya cikmamasi icin (veri
> gizliligi), ayrica gercek API anahtarina da artik gerek yok. Ek olarak
> **Google LangExtract** entegre edildi: `persons`/`organizations`
> alanlari artik mumkun oldugunda metinde birebir gecen, karakter konumu
> dogrulanmis degerler olarak cikariliyor.

---

## ÇALIŞTIRMA SIRASI (baştan sona, sırayla)

### 1 · Kur
```powershell
python -m pip install -r Agents/extraction_agent/requirements.txt
```
Bu, `pydantic`, `python-dotenv`, `openai` ve `langextract`'i kurar.
**Neden `pip` yerine `python -m pip`?** Birden fazla Python sürümü
kuruluysa (`3.11` ve `3.14` gibi), tek başına `pip` komutu yanlış/bozuk
bir sürüme denk gelebilir ("Fatal error in launcher"). `python -m pip`
her zaman o an çalıştırdığın `python` ile aynı ortama kurar.

### 2 · `.env` dosyasının var olduğundan emin ol
İçeriği bölüm 7'de -- yoksa oradan kopyala, `Agents/extraction_agent/.env`
olarak kaydet.

### 3 · Yerel Gemma 3 sunucusunu başlat (ayrı terminal, açık bırak)
```powershell
.\llama\llama-server.exe -m Inference\models\gemma-3.gguf --mmproj Inference\models\gemma-3-mmproj.gguf --host 0.0.0.0 --port 8092 --ctx-size 8192 --image-min-tokens 1024 --alias gemma-3-27b-it
```
`--mmproj` bayrağını unutma. `--alias` değeri `.env` içindeki
`EXTRACTION_LLM_MODEL`/`EXTRACTION_VLM_MODEL` ile eşleşmeli. Bu,
`classification_agent` ile **aynı paylaşılan sunucu/port** -- ikinci bir
sunucuya gerek yok.

### 4 · Agent'ı çalıştır (yeni terminal, proje kökünden)
```powershell
python -m Agents.extraction_agent.test_agent
```
⚠️ **Mutlaka proje kök dizininden** (`Agents` klasörünün göründüğü
yerden) çalıştır. `extraction_agent` klasörüne girip doğrudan
`python test_agent.py` çalıştırma -- relative import hatası verir.

### 5 · (Önerilir) Üç Agent'ı zincir halinde test et
`classification_agent` → `extraction_agent` → `validation_agent`
zincirinin gerçekten bağlı çalıştığını gösteren en iyi test:
```powershell
python -c "
from Agents.validation_agent.agent import ValidationAgent
from Agents.classification_agent.agent import ClassificationAgent
from Agents.extraction_agent.agent import ExtractionAgent
import json

state = {'ocr_result': {'full_text': 'Sayı: 2024/458 Ahmet Yılmaz, 0532 123 45 67, ahmet@example.com'}}
state = ClassificationAgent().run(state)
state = ExtractionAgent().run(state)
state = ValidationAgent().run(state)
print(json.dumps(state['validation'], ensure_ascii=False, indent=2))
"
```

---

## Sorun mu çıktı? (Troubleshooting -- gelişim sürecinde karşılaşılanlar)

| # | Hata / Belirti | Sebep | Çözüm |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'extraction_agent'` | Paket `Agents/` altında, proje kökünde değil | `python -m Agents.extraction_agent.test_agent` kullan (proje kökünden) |
| 2 | `ModuleNotFoundError: No module named 'pydantic'` | Kütüphane kurulu değil | `python -m pip install -r Agents/extraction_agent/requirements.txt` |
| 3 | `pip install ...` → *"Fatal error in launcher"* | Birden fazla Python sürümü, `pip` yanlış sürüme işaret ediyor | `python -m pip install ...` kullan |
| 4 | `.env` okunmuyor, LLM yanlış adrese gidiyor | `test_agent.py` çalıştırıldığında Python önce `Agents/extraction_agent/__init__.py`'yi (→`agent.py`→`config.py`) çalıştırır, `test_agent.py` en sona kalır -- `load_dotenv()` çok geç kalıyordu | `load_dotenv()` çağrısı `config.py`'nin en başına taşındı (config.py, env okuyan ilk dosya) |
| 5 | NER modeli yüklenemiyor (`No module named 'transformers'`) | `transformers`/`torch` kurulu değil (~2-3 GB) | Ekip kararı: ayrı NER modeli kurmak yerine `persons`/`organizations` aynı LLM çağrısından + LangExtract ile alınıyor. `EXTRACTION_NER_ENABLED=true` yapıp kurarsan yine aktif edilebilir |
| 6 | Gemma 3'e geçince LLM adımı sessizce atlanıyor, sonra timeout | **(a)** Kod gerçek API anahtarı zorunlu kılıyordu (Together AI için mantıklıydı, yerel sunucu için gereksiz); **(b)** `60s` timeout, CPU'da gözlemlenen gerçek gecikmeye (76-126 sn/çağrı) göre çok kısaydı | **(a)** anahtar yoksa `"not-needed"`'e düşecek şekilde düzeltildi; **(b)** timeout `180s`'ye yükseltildi |
| 7 | LangExtract `used: true` ama `persons`/`organizations` boş | Küçük kuantize (Q4) model CPU'da bazen bozuk/boş JSON parçası dönüyor -- LangExtract bunu sessizce atlıyor | **Bilinen sınırlama, çözülmedi.** Daha büyük/az-kuantize model ile tekrar denenmeli |
| 8 | `validation_agent` başarılı çalışmaları bile "eksik" işaretliyor | `extraction_agent` sadece `state["extraction_result"]` (ayrıntılı) yazıyordu, `validation_agent` ise `state["extraction"]` (düz format) okuyordu -- bu anahtar hiç yazılmıyordu | `agent.py`'ye `_to_unified_contract()` eklendi, artık `state["extraction"]` da yazılıyor (`classification_agent`'a da aynı şekilde `state["classification"]` eklendi) |

---

## Son Test Sonucu (gerçek, yerel Gemma 3 ile)

Sorun 6 (timeout) düzeltmesinden sonra alınan gerçek çıktı (mock değil):

```json
{
  "document": {
    "evrak_turu": "Şikayet",
    "tarih": { "value": "12.05.2024", "confidence": 0.95, "source": "regex" },
    "sayi": { "value": "2024/4521", "confidence": 0.85, "source": "regex" }
  },
  "entities": {
    "person": {
      "telefon": [{ "value": "0532 123 45 67", "confidence": 0.9, "source": "regex" }],
      "eposta": [{ "value": "ahmet.yilmaz@example.com", "confidence": 0.97, "source": "regex" }]
    }
  },
  "meta": { "success": true, "llm_used": true, "langextract_used": true, "errors": [], "warnings": [] }
}
```

**Kritik gözlem:** LLM/LangExtract çağrısı tamamen başarısız olsa bile
`meta.success: true` kalıyor ve regex katmanının bulduğu alanlar her
zaman doğru geliyor -- "fail-soft" tasarımı tam olarak çalışıyor.

---

## `.env` İçeriği (referans, güncel)

```env
EXTRACTION_LLM_ENABLED=true
EXTRACTION_LLM_BASE_URL=http://localhost:8092/v1
EXTRACTION_LLM_API_KEY=not-needed
EXTRACTION_LLM_MODEL=gemma-3-27b-it
EXTRACTION_LLM_TEMPERATURE=0.1
EXTRACTION_LLM_MAX_TOKENS=800
EXTRACTION_LLM_TIMEOUT=180
EXTRACTION_LLM_MAX_RETRIES=1

EXTRACTION_USE_LANGEXTRACT=true
EXTRACTION_LANGEXTRACT_PASSES=1
EXTRACTION_LANGEXTRACT_MAX_CHAR_BUFFER=4000

EXTRACTION_NER_ENABLED=false
EXTRACTION_NER_MODEL=akdeniz27/bert-base-turkish-cased-ner
EXTRACTION_NER_DEVICE=cpu
EXTRACTION_NER_MAX_CHARS=4000

EXTRACTION_VLM_ENABLED=true
EXTRACTION_VLM_BASE_URL=http://localhost:8092/v1
EXTRACTION_VLM_API_KEY=not-needed
EXTRACTION_VLM_MODEL=gemma-3-27b-it
EXTRACTION_VLM_TIMEOUT=180

EXTRACTION_CONF_THRESHOLD=0.55
```

---

## Beklenen `state` Anahtarları (Diğer Agent'larla Entegrasyon)

**Girdi (doğrulandı -- gerçek kod okunarak):**
| Anahtar | Format | Kaynak |
|---|---|---|
| `state["ocr_result"]` | `dict` (`{document_id, full_text, pages, ...}`) | `ocr_agent` |
| `state["classification_result"]` | ayrıntılı `dict` | `classification_agent` |
| `state["classification"]` | `{success, classification_confidence, document_type, status}` | `classification_agent` (kısa format) |
| `state["document_image_b64"]` *(isteğe bağlı)* | base64 string | imza/tablo varsa Vision için |

**Çıktı:**
| Anahtar | Format | Kim okuyor |
|---|---|---|
| `state["extraction_result"]` | ayrıntılı (yukarıdaki JSON gibi) | genel/debug |
| `state["extraction"]` | `{success, sender, date, address, phone, email}` | `validation_agent` |

---

## Klasör Yapısı ve Her Dosyanın Görevi

```
Agents/extraction_agent/
├── __init__.py         -> paketi dışarıya açar
├── agent.py             -> ana ExtractionAgent sınıfı
├── config.py             -> tüm ayarlar (eşikler, model adları, URL'ler)
├── models.py               -> Pydantic JSON şeması (çıktı formatı)
├── prompts.py                -> LLM için sistem + retry promptları
├── tools.py                    -> Regex / LLM / LangExtract / Vision araç sınıfları
├── test_agent.py                  -> hızlı manuel test scripti
├── requirements.txt                  -> Python bağımlılıkları
├── .env                                 -> ortam değişkenleri (gizli, git'e eklenmez)
└── README.md                               -> bu dosya
```

### `agent.py` -- Ana Sınıf
`ExtractionAgent(BaseAgent)`, `Agents/base/base_agent.py` kontratına
uyar (`run(state) -> state`) ve `agent_registry.py` üzerinden
`@register` ile kendini otomatik kaydeder. `run()` sırasıyla: state
içinden OCR çıktısını ve varsa sınıflandırma sonucunu **birden fazla
olası anahtar ismini deneyerek** okur (`ocr_result`/`ocr_output`,
`classification_result`/`classification`) -- takım arkadaşlarının
anahtar ismini değiştirmesi durumunda kırılmasın diye bilinçli bir
esneklik. Ardından dört aracı (regex, LLM, LangExtract, vision) sırayla
çalıştırır, hepsinin sonucunu `models.py`'deki standart şemaya göre
birleştirir, genel bir güven skoru hesaplar, ve **iki yere** yazar:
`state["extraction_result"]` (document/entities/request/vision/meta
içeren tam ayrıntılı yapı) ve `state["extraction"]` (validation_agent'ın
okuduğu düz/kısa format -- `success`, `sender`, `date`, `address`,
`phone`, `email`). Bu ikinci anahtar sonradan eklendi çünkü onsuz
`validation_agent` her zaman boş bir sözlük görüyor, başarılı çalışmaları
bile "eksik veri" olarak işaretliyordu (bkz. Sorun 8).

### `config.py` -- Ayarlar
Tüm eşik değerlerini ve model/URL ayarlarını ortam değişkenlerinden
(`.env`) okur. **En kritik detay:** `.env` dosyası bu dosyanın en
başında yüklenir (`load_dotenv(...)`) -- nedeni Sorun 4'te anlatıldı,
kısaca: hangi giriş noktasından (`test_agent.py`, Supervisor, vs.)
çalıştırılırsa çalıştırılsın, `config.py` her zaman ilk import edilen
dosya olduğu için `.env` yüklemesinin orada olması gerekiyor. İçerdiği
ayar grupları: `RegexConfig` (tarih/telefon/email/evrak-no
pattern'leri -- kural tabanlı, model gerektirmez), `NERConfig`
(varsayılan **kapalı**, bkz. Sorun 5), `LLMConfig` (Gemma 3 metin
bağlantı ayarları -- `base_url`, `model`, `timeout` -- artı LangExtract
ayarları: `use_langextract`, `langextract_extraction_passes`,
`langextract_max_char_buffer`), `VisionConfig` (Gemma 3 görsel bağlantı
ayarları), ve hepsini birleştiren üst-düzey `ExtractionAgentConfig`.

### `models.py` -- Çıktı Şeması (Pydantic)
Tüm agent'lar arası standart JSON formatını tanımlar
(`{document, entities, request, validation}` yapısına uygun --
`validation` kısmını bu agent değil, `validation_agent` doldurur). Her
tek değer bir `FieldValue` içinde tutulur: `{value, confidence, source,
char_start, char_end}`. `source` alanı değerin nereden geldiğini
gösterir: `"regex"`, `"llm"`, `"ner"` veya `"vision"`. `char_start`/
`char_end` -- sadece LangExtract, metinde birebir geçen bir değerin
karakter aralığını doğrulayabildiğinde doluyor (ör. bir kişi adı için);
regex/ner sonuçlarında veya LangExtract eşleşme bulamadığında `null`
kalıyor -- yani bu alanların dolu olması "bu değer metinde gerçekten var,
doğrulandı" anlamına geliyor, boş olması ise "ya bu yöntem konum takibi
yapmıyor ya da doğrulanamadı" demek, hatayı göstermiyor. `ExtractionMeta`
içinde ayrıca `success` (bu çalışma gerçekten denendi mi, yoksa OCR
metni boş olduğu için hiç başlamadı mı) ve `langextract_used` (grounded
çıkarım denendi mi) alanları var. Ana sınıflar: `DocumentInfo`
(tarih/sayı/konu), `Entities` (person + organization), `SemanticInfo`
(request_type/topic/intent/keywords), `VisionInfo`, `ExtractionMeta`.

### `prompts.py` -- LLM Promptları
Gemma 3'e gönderilen sistem promptu ve beklenen JSON şeması burada
tanımlı -- model-agnostik yazıldığı için Qwen'den Gemma'ya geçişte bu
dosyada tek satır değişmedi. Şema şu alanları içeriyor: `request_type`,
`topic`, `intent`, `keywords`, `missing_info`, ve **`persons`**/
**`organizations`** (bu ikisi, ayrı bir NER modeli kurmadan kişi/kurum
adlarını almak için sonradan eklendi, bkz. Sorun 5). Ayrıca, ilk deneme
başarısız olursa (geçersiz JSON veya düşük güven skoru) kullanılan daha
sıkı/net bir **retry (tekrar deneme) promptu** var.

### `tools.py` -- Araç Sınıfları (fail-soft tasarım)
Bu dosyadaki her araç, kendi başına asla pipeline'ı çökertmeyecek
şekilde yazıldı -- model indirilemezse, API key eksikse, veya LLM
geçersiz cevap dönerse, o katman sessizce atlanır ve
`meta.warnings`'e bir not eklenir.
- **`RegexExtractor`**: tarih, telefon, email, evrak/sayı numarası --
  kural tabanlı, %100 deterministik, hiçbir modele ihtiyaç duymaz.
- **`NEREngine`**: ayrı bir NER modeli (BERTurk) için altyapı, şu an
  devre dışı ama kod hazır dursun diye silinmedi -- ihtiyaç olursa tek
  bir env değişkeniyle geri açılabilir.
- **`LLMSemanticExtractor`**: Gemma 3'e OpenAI-uyumlu
  `/v1/chat/completions` çağrısı yapar (yerel `llama-server`, varsayılan
  port 8092), dönen metni güvenli şekilde JSON'a çevirir (` ```json `
  bloklarını temizler), düşük güven skorunda otomatik olarak daha sıkı
  bir promptla tekrar dener. `request_type`/`topic`/`intent`/`keywords`/
  `persons`/`organizations` üretir.
- **`LangExtractSemanticExtractor` [YENİ]**: Google LangExtract
  kütüphanesini kullanarak **sadece** `persons`/`organizations` için
  karakter-konumu doğrulanmış (grounded) çıkarım yapar -- aynı Gemma 3
  sunucusunu kullanır, ayrı model veya altyapı gerekmez. Neden sadece bu
  iki alan: LangExtract'ın gerçek değeri, metinde birebir geçen kelime
  gruplarını (span) orijinal metne hizalayabilmesinde; `request_type`/
  `topic` gibi sınıflandırma-tarzı alanlar zaten modelin kendi
  parafrazı/özeti olduğu için metinde birebir aranamıyor -- bu test
  edilip doğrulandı, grounded olmayan alanlarda LangExtract hiçbir fayda
  sağlamadan sadece ekstra bir çağrı ekliyor.
- **`HybridSemanticExtractor` [YENİ]**: `LLMSemanticExtractor` ile
  tamamen aynı `.extract()` sözleşmesini taşıyan bir sarmalayıcı
  (wrapper) -- önce normal LLM çağrısını olduğu gibi çalıştırır, sonra
  (eğer aktifse) `LangExtractSemanticExtractor` ile `persons`/
  `organizations` alanlarını grounded değerlerle **üzerine yazar**
  (override). LangExtract başarısız olursa veya kurulu değilse, sessizce
  eski (grounded olmayan) LLM sonucuna geri döner -- yani hiçbir zaman
  agent'ı bozmaz, en kötü ihtimalle eski davranışa eşdeğer çalışır.
  `agent.py`, artık `LLMSemanticExtractor` yerine doğrudan bunu
  kullanıyor.
- **`VisionFieldExtractor`**: sadece imza/tablo/el yazısı tespit
  edildiğinde ve bir görsel (base64) sağlandığında çalışır -- gereksiz
  yere her belgede görsel modele çağrı atmaz.

### `test_agent.py` -- Manuel Test
Sahte ama gerçekçi bir `state` oluşturur (rapordaki elektrik faturası
örneğiyle aynı metin), `ExtractionAgent.run()`'i çağırır ve sonucu JSON
olarak ekrana basar. `.env` dosyasını otomatik yükler. Hem pytest'e
gerek kalmadan hızlı doğrulama için, hem de yeni birine "bu nasıl
çalışıyor" sorusuna canlı bir cevap vermek için var.

### `requirements.txt` [YENİ]
`pydantic`, `python-dotenv`, `openai` (LLM/Vision için OpenAI-uyumlu
istemci), **`langextract`** (grounded entity extraction için). Daha
önce bu dosya hiç yoktu.

### `.env` -- Ortam Değişkenleri
Gerçek sunucu adresleri, model isimleri, eşik değerleri. **Bu dosya
git'e eklenmemeli** -- içinde özel bilgi olmasa bile, herkesin kendi
yerel ayarına göre değiştirebilmesi için `.gitignore`'a eklenmesi
önerilir.

---

## Yarışma İçin Ek Geliştirme Fikirleri

- **Precision/Recall/F1 scripti**: küçük bir ground-truth JSON seti
  hazırlayıp agent çıktısıyla karşılaştırmak -- LangExtract'ın
  `char_start`/`char_end` alanlarını da bu değerlendirmeye dahil etmek
  (grounded vs. grounded-olmayan doğruluk farkını ölçmek için).
- **Cache**: aynı metin (hash) için LLM sonuçlarını önbellekte tutup
  demo sırasında tekrar çağrıyı önlemek -- yerel CPU inference'in yavaş
  olduğu göz önüne alınırsa (bkz. Sorun 6) bu özellikle değerli.
- **Batch mod**: bir klasör dolusu OCR örneğini toplu işleyip performans
  raporu üreten bir fonksiyon.
- **LangExtract güvenilirliği**: Sorun 7'nin kökten çözümü için daha
  büyük/az-kuantize bir model ile ablation testi.
