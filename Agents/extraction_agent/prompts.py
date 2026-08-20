"""
----------
Extraction_agent'ın LLM katmanı için istem şablonları (Qwen2.5 Instruct).

İki istem özellikle sağlanmıştır (rapor bölümü 10 - Yeniden Dene ve Doğrulama): İlk geçiş düşük güvenilirlik/hatalı JSON döndürürse,
tools.LLMSemanticExtractor, daha katı olan ve örnek bir uygulama içeren ALT_EXTRACTION_PROMPT ile bir kez daha dener.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\

Sema:
{
  "request_type": string | null,   // "Sikayet" | "Bilgi Talebi" | "Basvuru" | "Ihbar" | "Diger"
  "topic": string | null,          // konunun kisa basligi (max 8 kelime)
  "intent": string | null,         // kullanicinin gercek talebi, tek cumle
  "keywords": string[],            // 3-6 anahtar kelime
  "missing_info": string[],        // eksik/belirsiz kalan bilgiler, varsa
  "persons": string[],             // metinde gecen KISI ad-soyadlari (tam isim)
  "organizations": string[],       // metinde gecen KURUM/MUDURLUK isimleri
  "confidence": number             // 0.0-1.0 arasi, kendi guven skorun
}

Kurallar:
- Metinde olmayan bilgi UYDURMA. Emin degilsen null/bos liste birak ve missing_info'ya ekle.
- "persons": sadece gercek kisi adlari (orn. "Ahmet Yılmaz"), unvan/roller ("Sayın Yetkili" gibi) dahil etme.
- "organizations": resmi kurum, mudurluk, sirket isimleri (orn. "Enerji Müdürlüğü").
- confidence: metin acik ve tek anlamliysa yuksek (0.8+), belirsiz/kisa/karisiksa dusuk (<0.5) ver.
- Cikti SADECE JSON objesi olmali.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Asagidaki belge metnini analiz et ve semaya uygun JSON uret.

{classification_hint_block}
Belge metni:
\"\"\"
{document_text}
\"\"\"
"""

# Stricter retry prompt - used when the first attempt fails to parse or
# comes back with confidence below the configured threshold.
ALT_EXTRACTION_PROMPT_TEMPLATE = """\
Ornek (format referansi icin, iceriği kopyalama):
Girdi: "Elektrik faturam beklediğimden yüksek geldi. İncelenmesini istiyorum. Ahmet Yılmaz, Enerji Müdürlüğü'ne başvurmuştur."
Cikti: {{"request_type": "Sikayet", "topic": "Elektrik faturasi", \
"intent": "Fatura inceleme talebi", "keywords": ["elektrik", "fatura", "inceleme"], \
"missing_info": [], "persons": ["Ahmet Yılmaz"], "organizations": ["Enerji Müdürlüğü"], "confidence": 0.9}}

{classification_hint_block}
Belge metni:
\"\"\"
{document_text}
\"\"\"

Sadece JSON don:
"""


def build_extraction_prompt(document_text: str, classification_hint: str | None, retry: bool = False) -> str:
    hint_block = f"Belge turu (Classification Agent): {classification_hint}\n" if classification_hint else ""
    template = ALT_EXTRACTION_PROMPT_TEMPLATE if retry else EXTRACTION_PROMPT_TEMPLATE
    return template.format(classification_hint_block=hint_block, document_text=document_text.strip()[:6000])


# --- Vision (Qwen-VL) prompt - report section 7 ---
VISION_SYSTEM_PROMPT = """\
Sen resmi bir evrak gorselini inceleyen bir asistansin. Sadece asagidaki \
JSON semasina uygun cikti ver, baska metin ekleme:

{
  "has_signature": boolean,
  "has_handwriting": boolean,
  "has_stamp": boolean,
  "has_table": boolean,
  "form_fields": object   // gorselde doldurulmus form alanlari varsa key:value olarak
}
"""

VISION_USER_PROMPT = "Bu evrak gorselini incele ve semaya uygun JSON don."