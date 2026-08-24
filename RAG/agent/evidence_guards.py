"""Dar kapsamlı, metinde açıkça bulunan hukukî olgular için güvenlik katmanı."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

from RAG.retriever.text_utils import fold_turkish, tokenize


_HOUR_QUESTION = re.compile(r"\b(kac|hangi)\s+saat\b|\bsaat\s+icinde\b")
_TRAVEL_HOUR = re.compile(
    r"\b(\d{1,3})\s*saat\s+icin(?:de)?\s+yola\s+cik",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_LAW_NUMBER = re.compile(r"\b(\d{3,5})\s+sayili\s+kanun\b")
_KHK_NUMBER = re.compile(r"\bkhk\s*[-/]?\s*(\d{2,4})\b", re.IGNORECASE)
_AYM_ANNULMENT = re.compile(
    r"anayasa\s+mahkemesinin\s+(\d{1,2})/(\d{1,2})/(\d{4})\s+tarihli\s+ve\s+"
    r"e\.?\s*[:.]?\s*(\d{4}/\d+)\s*,?\s*k\.?\s*[:.]?\s*(\d{4}/\d+)",
    re.IGNORECASE,
)
_DATE_DIFFERENCE = re.compile(r"\b(?:zaman|sure)\s+farki\b|\barasindaki\s+fark\b")
_CROSS_LAW_REFERENCE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})\s+tarih(?:li)?\s+ve\s+"
    r"(\d{3,5})\s+sayili\s+(.{2,80}?)\s+kanun(?:u|un|una|unun|dan|da)?\b",
    re.IGNORECASE,
)
_KHK_EFFECTIVE_DATE = re.compile(
    r"\bkhk\s*[-/]?\s*(\d{2,4})\b.{0,280}?(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
    re.IGNORECASE,
)


def _plain(value: object) -> str:
    return fold_turkish(str(value or "")).casefold()


def _display_law(source: Dict[str, object]) -> str:
    number = str(source.get("law_number") or "").strip()
    name = str(source.get("law_name") or "Kanun").replace("_", " ").strip()
    if number and number != "unknown":
        name = re.sub(rf"^{re.escape(number)}\s*", "", name).strip()
        return f"{number} sayılı {name}"
    return name


def _date_key(value: str) -> Optional[tuple[int, int, int]]:
    match = _DATE.search(value)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def _format_date(value: str) -> str:
    """PDF'deki değişken tarih gösterimini tek bir kullanıcı biçimine dönüştürür."""
    parsed = _date_key(value)
    if not parsed:
        return value
    day, month, year = parsed
    return f"{day:02d}/{month:02d}/{year}"


def _display_reference_title(question: str, normalized_title: str) -> str:
    """Sorgudaki Türkçe yazımı korur; arama için katlanmış başlık kullanılır."""
    title_terms = tokenize(normalized_title, min_len=2)
    if not title_terms:
        return normalized_title
    word_count = len(title_terms)
    for match in re.finditer(
        r"(.{0,100}?)\s+Kanun(?:unun|una|un|u|dan|da)\b", question, re.IGNORECASE
    ):
        words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", match.group(1))
        candidate = " ".join(words[-word_count:])
        if all(term in _plain(candidate) for term in title_terms):
            return candidate
    return normalized_title


def direct_cross_law_reference_answer(question: str, sources: Iterable[Dict[str, object]]) -> Optional[str]:
    """Bir maddede açıkça atıf yapılan kanunun numara/tarih bilgisini sunar.

    Bu yol, soru iki bağımsız olgu istese bile LLM'in yalnız değişiklik
    cetveline bakıp madde içindeki kanun atfını kaçırmasını engeller.
    """
    normalized_question = _plain(question)
    asks_reference_identity = any(term in normalized_question for term in (
        "kabul tarih", "kanun numara", "yasa numara",
    ))
    if not asks_reference_identity:
        return None

    reference: Optional[tuple[str, str, str, str]] = None
    requested_khk = _KHK_NUMBER.search(normalized_question)
    khk_effective: Optional[tuple[str, str]] = None

    for source in sources:
        text = _plain(source.get("text"))
        label = str(source.get("label") or "S1")
        for match in _CROSS_LAW_REFERENCE.finditer(text):
            date_raw, law_number, law_title = match.groups()
            title_terms = tokenize(law_title, min_len=3)
            # Aynı maddede birden fazla kanun geçerse yalnız soruda adı geçen
            # kanun tercih edilir; başlık tek kısa kelimeden oluşuyorsa esnek kalır.
            if title_terms and not any(term in normalized_question for term in title_terms):
                continue
            reference = (law_number, law_title.strip(), _format_date(date_raw), label)
            break
        if reference:
            break

    if requested_khk:
        requested_number = requested_khk.group(1)
        for source in sources:
            match = _KHK_EFFECTIVE_DATE.search(_plain(source.get("text")))
            if not match or match.group(1) != requested_number:
                continue
            khk_effective = (_format_date(match.group(2)), str(source.get("label") or "S1"))
            break

    if not reference:
        return None

    law_number, law_title, acceptance_date, reference_label = reference
    law_title = _display_reference_title(question, law_title)
    lines = [
        f"Soruda belirtilen {law_title} Kanunu, {acceptance_date} kabul tarihli "
        f"{law_number} sayılı Kanundur. [{reference_label}]"
    ]
    if requested_khk and khk_effective:
        date_raw, label = khk_effective
        lines.append(
            f"KHK/{requested_khk.group(1)} sayılı Kararname ise {date_raw} tarihinde "
            f"yürürlüğe girmiştir. [{label}]"
        )
    return "\n\n".join(lines)


def direct_travel_duration_answer(question: str, sources: Iterable[Dict[str, object]]) -> Optional[str]:
    """Yalnız açık 'X saat içinde yola çıkma' hükmünü deterministik sunar.

    Aynı maddede veya komşu pasajda geçen farklı süreleri bir aralığa dönüştürmez.
    Bu, LLM'in 6 saatlik seferberlik kuralını başka koşuldaki süreyle karıştırmasını önler.
    """
    normalized_question = _plain(question)
    if not _HOUR_QUESTION.search(normalized_question) or "yola cik" not in normalized_question:
        return None

    best: Optional[Dict[str, object]] = None
    best_hours: Optional[str] = None
    best_score = -1
    for source in sources:
        text = _plain(source.get("text"))
        match = _TRAVEL_HOUR.search(text)
        if not match:
            continue
        score = sum(term in text for term in ("seferberlik", "ilan", "sefer gorev", "yukuml"))
        # Kullanıcının sorusundaki koşulların aynı pasajda geçmesi zorunludur.
        if "seferberlik" in normalized_question and "seferberlik" not in text:
            continue
        if "sefer gorev" in normalized_question and "sefer gorev" not in text:
            continue
        if score > best_score:
            best, best_hours, best_score = source, match.group(1), score

    if not best or not best_hours:
        return None

    article = str(best.get("article_number") or best.get("article_no") or "-")
    label = str(best.get("label") or "S1")
    answer = (
        f"{_display_law(best)} kapsamındaki Madde {article} hükmüne göre, "
        f"sefer görevi olan yükümlü seferberlik ilanı hâlinde ilan saatinden itibaren "
        f"{best_hours} saat içinde yola çıkmakla yükümlüdür. [{label}]"
    )

    # Soru bir değişikliğin yürürlük tarihini de içeriyorsa, yalnız cetvel/ledger
    # kaynağında aynı tarih gerçekten bulunduğunda ikinci olguyu ekle.
    asked_date = _date_key(normalized_question)
    requested_laws = set(_LAW_NUMBER.findall(normalized_question))
    if asked_date and requested_laws:
        for source in sources:
            text = _plain(source.get("text"))
            if not any(re.search(rf"\b{re.escape(law)}\b", text) for law in requested_laws):
                continue
            if _date_key(text) != asked_date:
                continue
            source_label = str(source.get("label") or "S1")
            answer += (
                f" {next(iter(requested_laws))} sayılı Kanunla yapılan ilgili değişikliğin "
                f"yürürlük tarihi {asked_date[0]:02d}/{asked_date[1]:02d}/{asked_date[2]}’dir. "
                f"[{source_label}]"
            )
            break
    return answer


def direct_constitutional_annulment_answer(question: str, sources: Iterable[Dict[str, object]]) -> Optional[str]:
    """Anayasa Mahkemesi iptal kararının tarih, E. ve K. bilgisini aynen verir."""
    normalized_question = _plain(question)
    intent_terms = ("khk", "anayasa mahkem", "yuksek mahk", "iptal", "gecersiz", "dava dosya", "karar kayit")
    if not any(term in normalized_question for term in intent_terms):
        return None

    requested_khks = set(_KHK_NUMBER.findall(normalized_question))
    for source in sources:
        text = _plain(source.get("text"))
        if requested_khks and not any(re.search(rf"\bkhk\s*[-/]?\s*{re.escape(number)}\b", text) for number in requested_khks):
            continue
        match = _AYM_ANNULMENT.search(text)
        if not match:
            continue
        day, month, year, case_no, decision_no = match.groups()
        label = str(source.get("label") or "S1")
        return (
            f"İlgili hüküm, Anayasa Mahkemesinin {int(day):02d}/{int(month):02d}/{year} tarihli "
            f"E.{case_no}, K.{decision_no} sayılı kararıyla iptal edilmiştir. [{label}]"
        )
    return None


def court_date_comparison_is_incomplete(question: str, sources: Iterable[Dict[str, object]]) -> bool:
    """Mahkeme kararı ile değişiklik tarihini kıyaslamak için iki açık tarih ister.

    Karar tarihi, kararın yürürlüğe girdiği tarih değildir. Bu ayrım belgede
    açıkça yazılmadığında LLM'in tarih farkı hesaplaması hukukî bir tahmin olur.
    """
    normalized_question = _plain(question)
    if not _DATE_DIFFERENCE.search(normalized_question):
        return False
    if not any(term in normalized_question for term in ("anayasa mahkem", "yuksek mahk", "iptal karari")):
        return False

    texts = [_plain(source.get("text")) for source in sources]
    has_amendment_date = any(
        ("degisiklik cetveli" in text or "degistiren duzenleme" in text)
        and bool(_DATE.search(text))
        for text in texts
    )
    has_court_effective_date = any(
        "anayasa mahkem" in text
        and bool(_DATE.search(text))
        and any(term in text for term in ("yururluge gir", "yururluk tarih", "resmi gazete"))
        for text in texts
    )
    return not (has_amendment_date and has_court_effective_date)


def incomplete_court_date_comparison_answer(
    question: str, sources: Iterable[Dict[str, object]]
) -> Optional[str]:
    """Eksik iki-tarih sorusunda bulunan somut bilgiyi kaybetmeden sunar."""
    source_list = list(sources)
    if not court_date_comparison_is_incomplete(question, source_list):
        return None

    normalized_question = _plain(question)
    amendment_numbers = re.findall(r"\b(\d{3,5})\s+sayili\s+kanun\b", normalized_question)
    requested_article = re.search(r"\b(\d+[a-z]?)\s*\.\s*madden", normalized_question)
    lines: list[str] = []

    for source in source_list:
        text = _plain(source.get("text"))
        label = str(source.get("label") or "S1")
        date = _DATE.search(text)
        if (
            date
            and ("degisiklik cetveli" in text or "degistiren duzenleme" in text)
            and (not amendment_numbers or any(number in text for number in amendment_numbers))
        ):
            day, month, year = date.groups()
            lines.append(
                f"- {amendment_numbers[0] if amendment_numbers else 'İlgili'} sayılı düzenleme için "
                f"doğrulanabilen yürürlük tarihi {int(day):02d}/{int(month):02d}/{year}’dir. [{label}]"
            )
            break

    if requested_article:
        article = requested_article.group(1)
        for source in source_list:
            if str(source.get("article_number") or source.get("article_no") or "") != article:
                continue
            text = _plain(source.get("text"))
            repeal = re.search(r"mulga\s*:\s*(\d{1,2}/\d{1,2}/\d{4})\s*[-–]\s*(\d{2,5})/(\d+)\s*md", text)
            if repeal:
                date_raw, instrument, provision = repeal.groups()
                lines.append(
                    f"- Madde {article} metninde “Mülga: {date_raw}-{instrument}/{provision} md.” kaydı yer alır. "
                    f"[{source.get('label') or 'S1'}]"
                )
            break

    if not lines:
        return None
    return (
        "Kaynaklarda doğrulanabilen bilgiler:\n" + "\n".join(lines) + "\n\n"
        "Ancak kaynaklarda, soruda belirtilen Anayasa Mahkemesi kararının Madde "
        f"{requested_article.group(1) if requested_article else '-'} için yürürlüğe giriş tarihi açıkça yer almıyor. "
        "Bu nedenle iki tarih arasındaki fark güvenilir biçimde hesaplanamaz."
    )
