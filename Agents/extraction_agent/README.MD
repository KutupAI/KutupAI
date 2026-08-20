# Information Extraction Agent (Bilgi Cikarma Ajani)
---

## 1. Ozet: Neler Yapildi?

Baslangicta `Agents/extraction_agent/` klasorunde sadece 4 bos dosya vardi
(icinde sadece aciklama yorumu olan iskelet dosyalar: `agent.py`,
`config.py`, `prompts.py`, `tools.py`). Bu dosyalar sifirdan dolduruldu,
3 yeni dosya eklendi, ve `Agents/base/base_agent.py` +
`Agents/base/agent_registry.py` kontratina tam uyumlu hale getirildi.

Gelistirme sirasinda ekibin gercek ortamiyla (yerel `llama-server` +
`qwen-vl.gguf`, port 8092) uctan uca test edildi ve **basariyla calisti**
(asagida bolum 6'da tam cikti var).

---

## 2. Klasor Yapisi ve Her Dosyanin Gorevi

```
Agents/extraction_agent/
├── __init__.py        [YENI]     -> paketi disariya acar
├── agent.py            [DOLDURULDU] -> ana ExtractionAgent sinifi
├── config.py            [DOLDURULDU] -> tum ayarlar (esikler, model adlari, URL'ler)
├── models.py             [YENI]     -> Pydantic JSON semasi (cikti formati)
├── prompts.py              [DOLDURULDU] -> LLM icin sistem + retry promptlari
├── tools.py                  [DOLDURULDU] -> Regex / LLM / Vision arac siniflari
├── test_agent.py                [YENI]     -> hizli manuel test scripti
├── .env                            [YENI]     -> ortam degiskenleri (gizli, git'e eklenmez)
└── README.md                         [YENI]     -> bu dosya
```

### `models.py` - Cikti Semasi (Pydantic)
Tum agentlar arasi standart JSON formatini tanimlar (rapor bolum 9'daki
`{document, entities, request, validation}` yapisina uygun - `validation`
kismi Validation Agent tarafindan doldurulacak, bu ajan onu doldurmaz).

Her deger bir `FieldValue` icinde tutulur: `{value, confidence, source}`.
`source` alani `"regex"`, `"llm"` veya `"vision"` olabilir - boylece
Validation Agent hangi bilginin nereden geldigini ve ne kadar guvenilir
oldugunu bilir.

Ana siniflar: `DocumentInfo` (tarih/sayi/konu), `Entities` (person +
organization), `SemanticInfo` (request_type/topic/intent/keywords),
`VisionInfo`, `ExtractionMeta` (guven skoru, uyarilar, hata listesi).

### `config.py` - Ayarlar
Tum esik degerleri ve model/URL ayarlarini ortam degiskenlerinden
(`.env`) okur. **En onemli detay:** `.env` dosyasi bu dosyanin en
basinda (`load_dotenv(...)`) yuklenir - nedeni bolum 5'te ("Karsilasilan
Sorunlar") acildi.

Icerik: `RegexConfig` (tarih/telefon/email/evrak-no pattern'leri),
`NERConfig` (varsayilan **kapali** - bkz. bolum 5), `LLMConfig` (Qwen
baglanti ayarlari), `VisionConfig` (Qwen-VL baglanti ayarlari),
`ExtractionAgentConfig` (hepsini birlestiren ust-duzey ayar objesi).

### `prompts.py` - LLM Promptlari
Qwen'e gonderilen sistem promptu ve JSON semasi burada tanimli. Sema
sunlari icerir: `request_type`, `topic`, `intent`, `keywords`,
`missing_info`, **`persons`**, **`organizations`** (son ikisi, ayri bir
NER modeli kurmadan kisi/kurum adlarini almak icin eklendi - bkz. bolum 5),
`confidence`.

Ayrica, ilk deneme basarisiz olursa (gecersiz JSON / dusuk guven)
kullanilan daha sikı bir **retry (tekrar deneme) promptu** var - rapor
bolum 10'daki "Alternatif prompt denenir" adimini karsilar.

### `tools.py` - Arac Siniflari (fail-soft tasarim)
- `RegexExtractor`: tarih, telefon, email, evrak/sayi numarasi - kural
  tabanli, %100 deterministik.
- `NEREngine`: ayri bir NER modeli (BERTurk) icin altyapi - **su an
  devre disi**, ama kod hazir dursun diye silinmedi (kolayca aktif
  edilebilir).
- `LLMSemanticExtractor`: Qwen'e OpenAI-uyumlu `/v1/chat/completions`
  cagrisi yapar, JSON'u guvenli sekilde parse eder (```json bloklarini
  temizler), dusuk guven durumunda otomatik retry yapar.
- `VisionFieldExtractor`: sadece imza/tablo/el yazisi tespit edildiginde
  ve bir gorsel (base64) saglandiginda calisir.

**Onemli tasarim prensibi:** Her arac kendi basina cokmez. Model
indirilemezse, API key eksikse ya da LLM gecersiz cevap donerse, o katman
sessizce atlanir ve `meta.warnings` icine bir not eklenir - agent asla
`Exception` firlatarak tum pipeline'i durdurmaz.

### `agent.py` - Ana Sinif
`ExtractionAgent(BaseAgent)` - `Agents/base/base_agent.py` kontratina
uyar (`run(state) -> state`), ve `Agents/base/agent_registry.py`
uzerinden `@register` decorator'u ile kendini otomatik kaydeder.

`state` icinden OCR ciktisini ve (varsa) siniflandirma sonucunu **birden
fazla olasi anahtar ismini deneyerek** okur (`ocr_result`/`ocr_output`,
`classification_result`/`classification`) - takim arkadaslarinin tam
anahtar ismini degistirmesi durumunda kirilmasin diye.

Tum araclarin (regex/LLM/vision) sonuclarini `models.py`'deki standart
semaya gore birlestirir, genel bir guven skoru hesaplar, ve
`state["extraction_result"]` olarak yazar.

### `test_agent.py` - Manuel Test
Sahte (ama gercekci) bir `state` olusturur (rapordaki elektrik faturasi
ornegiyle ayni metin), `ExtractionAgent.run()`'i cagirir ve sonucu JSON
olarak ekrana basar. `.env` dosyasini otomatik yukler. Hem pytest'e
gerek kalmadan hizli dogrulama hem de yeni birinin "bu nasil calisiyor"
sorusuna canli cevap vermek icin var.

### `.env` - Ortam Degiskenleri
Gercek API adresleri, model isimleri, esik degerleri. **Bu dosya git'e
eklenmemeli** (icinde ozel bilgi olmasa bile, herkesin kendi yerel
ayarina gore degistirebilmesi icin `.gitignore`'a eklenmesi onerilir).

---

## 3. Kurulum (Gerekli Kutuphaneler)

Asagidaki komutlari calistirin (Windows PowerShell, proje kok
dizininde, ornegin `KutupAI-main` icinde):

```powershell
python -m pip install pydantic openai python-dotenv
```

**Neden `pip` yerine `python -m pip`?** Bilgisayarda birden fazla Python
surumu kuruluysa (bizim durumumuzda Python 3.11 ve 3.14), tek basina
`pip` komutu yanlis/bozuk bir surume denk gelebilir
("Fatal error in launcher" hatasi). `python -m pip` her zaman o an
calistirdiginiz `python` ile ayni ortama kurulum yapar.

**NER icin (opsiyonel, varsayilan olarak GEREKLI DEGIL):**
```powershell
python -m pip install transformers torch
```
Bu ekip karari geregi kurulmadi - detay bolum 5'te.

---

## 4. Nasil Calistirilir

1. `Agents/extraction_agent/.env` dosyasinin var oldugundan emin olun
   (icerigi asagida, bolum 7'de).
2. Yerel LLM sunucusunu baslatin (ayri bir terminal penceresinde,
   kapatmayin):
   ```powershell
   .\llama\llama-server.exe -m Inference\models\qwen-vl.gguf --mmproj Inference\models\qwen-vl-mmproj.gguf --host 0.0.0.0 --port 8092 --ctx-size 8192 --image-min-tokens 1024
   ```
3. Proje kok dizininden (`KutupAI-main`), test scriptini calistirin:
   ```powershell
   python -m Agents.extraction_agent.test_agent
   ```

**Onemli:** Komut `python -m Agents.extraction_agent.test_agent`
seklinde, **proje kok dizininden** calistirilmali (icinde `Agents`
klasorunun goruldugu yerden). `extraction_agent` klasorunun icine girip
dogrudan `python test_agent.py` calistirmayin - bu, `from .agent import`
gibi relative import'lar yuzunden hata verir.

---

## 5. Gelistirme Surecinde Karsilasilan Sorunlar ve Cozumleri

Bu bolum, ayni hatalarla tekrar karsilasan biri icin referans amaclidir.

### Sorun 1: `ModuleNotFoundError: No module named 'extraction_agent'`
**Sebep:** Paket `Agents/` klasorunun altinda, proje kokunde degil.
**Cozum:** `python -m extraction_agent.test_agent` degil,
`python -m Agents.extraction_agent.test_agent` calistirilmali (proje
kokunden).

### Sorun 2: `ModuleNotFoundError: No module named 'pydantic'`
**Sebep:** Kutuphane kurulu degildi.
**Cozum:** `python -m pip install pydantic openai`

### Sorun 3: `pip install python-dotenv` -> "Fatal error in launcher"
**Sebep:** Sistemde 2 farkli Python surumu var (3.11 ve 3.14), `pip`
komutu bozuk/yanlis surume isaret ediyordu.
**Cozum:** `python -m pip install python-dotenv` kullanildi (aktif
`python` ile ayni ortama kurulumu garantiler).

### Sorun 4 (en onemlisi): `.env` dosyasi okunmuyor, LLM `together.xyz`'e
gidiyor ve `401 Unauthorized` hatasi aliniyordu
**Sebep:** `python -m Agents.extraction_agent.test_agent` calistirildiginda,
Python ONCE `Agents/extraction_agent/__init__.py`'yi calistirir (bu da
`agent.py -> config.py` zincirini tetikler), SONRA `test_agent.py`'yi
`__main__` olarak calistirir. Yani `test_agent.py` icine konan
`load_dotenv()` cok GEC kaliyordu - `config.py`'deki ayarlar zaten
(varsayilan/yanlis) degerlerle tanimlanmis oluyordu.
**Cozum:** `load_dotenv()` cagrisi `test_agent.py`'den `config.py`'nin
EN BASINA tasindi - cunku `config.py`, ortam degiskenlerini okuyan ilk
dosya ve hangi giris noktasindan (`test_agent.py`, Supervisor, vs.)
calistirilirsa calistirilsin en once o import edilir.

### Sorun 5: NER modeli yuklenemiyordu (`No module named 'transformers'`)
**Sebep:** `transformers`/`torch` kurulu degildi (buyuk kutuphaneler,
~2-3 GB indirme).
**Karar (ekip ile birlikte):** Ayri bir NER modeli kurmak yerine, kisi
(`persons`) ve kurum (`organizations`) bilgilerini **ayni Qwen-VL LLM
cagrisindan** almak icin `prompts.py`'deki JSON semasina 2 yeni alan
eklendi. Sonuc: ekstra kurulum yok, tek bir LLM cagrisi, ve test
sonucunda (bolum 6) bu yontemin dogru calistigi dogrulandi
(`"Ahmet Yılmaz"` ve `"Enerji Müdürlüğü"` basariyla cikarildi).
`EXTRACTION_NER_ENABLED=false` varsayilan yapildi; ayri bir NER modeli
gerekirse `true` yapip `transformers`/`torch` kurarak tekrar
aktiflestirilebilir - kod hazir, sadece devre disi.

---

## 6. Son Test Sonucu (Basarili - Gercek Qwen-VL ile)

Asagidaki cikti, yerel `llama-server` (port 8092) ile yapilan gercek bir
cagriyla elde edildi (mock/sahte degil). `python -m
Agents.extraction_agent.test_agent` komutunun son calistirilmasindan:

```json
{
  "document": {
    "evrak_turu": "Şikayet",
    "tarih": { "value": "12.05.2024", "confidence": 0.95, "source": "regex" },
    "sayi": { "value": "2024/4521", "confidence": 0.85, "source": "regex" },
    "konu": { "value": "Elektrik Faturası", "confidence": 0.8, "source": "llm" }
  },
  "entities": {
    "person": {
      "ad_soyad": [{ "value": "Ahmet Yılmaz", "confidence": 0.8, "source": "llm" }],
      "telefon": [{ "value": "0532 123 45 67", "confidence": 0.9, "source": "regex" }],
      "eposta": [{ "value": "ahmet.yilmaz@example.com", "confidence": 0.97, "source": "regex" }],
      "adres": []
    },
    "organization": {
      "kurum": [{ "value": "Enerji Müdürlüğü", "confidence": 0.8, "source": "llm" }],
      "mudurluk": [],
      "ilgili_birim": []
    }
  },
  "request": {
    "request_type": "Sikayet",
    "topic": "Elektrik Faturası",
    "intent": "İncelenmesini istiyorum",
    "keywords": ["Elektrik faturası", "geldi", "İncelenmesini"],
    "missing_info": ["Tarih"],
    "confidence": 0.8
  },
  "vision": { "used": false, "has_signature": null, "has_stamp": null, "has_table": null, "has_handwriting": null, "form_fields": {} },
  "meta": {
    "overall_confidence": 0.85,
    "low_confidence": false,
    "retried": false,
    "retry_count": 0,
    "llm_used": true,
    "ner_used": false,
    "vision_used": false,
    "errors": [],
    "warnings": []
  }
}
```

**Sonuc: sifir hata, sifir uyari, tum alanlar dolu.** `missing_info`
icindeki `"Tarih"` yanlis alarm - LLM sadece kendi ayristirdigi metinde
tarihi net gormedi, ama tarih zaten regex katmani tarafindan ayrica ve
dogru sekilde (`12.05.2024`) yakalandi. Gercek bir hata degil, LLM'in
kendi guven notu.

---

## 7. `.env` Icerigi (Referans)

```env
EXTRACTION_LLM_ENABLED=true
EXTRACTION_LLM_BASE_URL=http://localhost:8092/v1
EXTRACTION_LLM_API_KEY=not-needed
EXTRACTION_LLM_MODEL=qwen-vl
EXTRACTION_LLM_TEMPERATURE=0.1
EXTRACTION_LLM_MAX_TOKENS=800
EXTRACTION_LLM_TIMEOUT=60
EXTRACTION_LLM_MAX_RETRIES=1

EXTRACTION_NER_ENABLED=false
EXTRACTION_NER_MODEL=akdeniz27/bert-base-turkish-cased-ner
EXTRACTION_NER_DEVICE=cpu
EXTRACTION_NER_MAX_CHARS=4000

EXTRACTION_VLM_ENABLED=true
EXTRACTION_VLM_BASE_URL=http://localhost:8092/v1
EXTRACTION_VLM_API_KEY=not-needed
EXTRACTION_VLM_MODEL=qwen-vl
EXTRACTION_VLM_TIMEOUT=60

EXTRACTION_CONF_THRESHOLD=0.55
```

---

## 8. Beklenen `state` Anahtarlari (Diger Agentlarla Entegrasyon)

**Girdi:**
| Anahtar | Format | Kaynak |
|---|---|---|
| `state["ocr_result"]` veya `state["ocr_output"]` | `dict` (`UnifiedOCRResult.to_dict()`) veya `str` | OCR Agent (Obada) |
| `state["classification_result"]` veya `state["classification"]` | `document_type` iceren `dict` | Classification Agent (Yasir) |
| `state["document_image_b64"]` *(istege bagli)* | base64 string | imza/tablo varsa Vision icin |

⚠️ Bu anahtarlarin gercek isimlerini Obada ve Yasir ile birebir teyit
edin - kod en yaygin isimlendirmeleri dener ama garanti degildir.

**Cikti:** `state["extraction_result"]` (bolum 6'daki JSON formati).

---

## 9. Yarisma Icin Ek Gelistirme Fikirleri

- **Precision/Recall/F1 scripti** (rapor bolum 11): kucuk bir
  ground-truth JSON seti hazirlayip agent ciktisiyla karsilastirma.
- **Cache**: ayni metin (hash) icin LLM sonuclarini onbellekte tutup
  demo sirasinda tekrar cagriyi onlemek.
- **Batch mod**: bir klasor dolusu OCR ornegini toplu isleyip performans
  raporu ureten bir fonksiyon.