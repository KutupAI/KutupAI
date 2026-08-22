"""
knowledge_base.py
==================

Structured routing knowledge base: a list of Department nodes plus a BM25
index built once over their descriptive corpora.

The seed data below models a generic Turkish public-institution hierarchy
(institution -> presidency/general directorate -> department -> unit ->
authority/makam) so the pipeline is runnable and testable out of the box.
Replace `_SEED_DEPARTMENTS` with real KutupAI department data in
production; nothing else in the module needs to change.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import Department
from .tools import BM25Index

_INSTITUTION = "Kutup Kurumu"
_PRESIDENCY = "Kutup Kurumu Genel Müdürlüğü"

_SEED_DEPARTMENTS: List[Department] = [
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="İnsan Kaynakları Daire Başkanlığı",
        unit="Personel İşlemleri Şube Müdürlüğü",
        authority_level="Daire Başkanı",
        makam="İnsan Kaynakları Daire Başkanlığı Makamı",
        responsibilities=[
            "personel atama işlemleri", "izin işlemleri", "emeklilik işlemleri",
            "kadro ve özlük dosyası yönetimi", "disiplin süreçleri (idari)",
            "hizmet içi eğitim planlaması",
        ],
        handled_topics=["personel", "atama", "izin", "emeklilik", "özlük", "kadro", "nakil", "terfi"],
        keywords=["personel", "atama", "izin", "kadro", "özlük dosyası", "emeklilik", "nakil", "terfi"],
        entities=["Personel Daire Başkanlığı", "İnsan Kaynakları"],
        legal_authority=["657 sayılı Devlet Memurları Kanunu"],
        required_documents=["personel bilgi formu", "atama onayı"],
        excluded_topics=["hukuki itiraz", "dava", "yargı süreci"],
        routing_rules=["personel atama talebi", "izin talebi", "emeklilik talebi"],
        channel_hint=None,
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Hukuk Müşavirliği",
        unit="Dava Takip Şube Müdürlüğü",
        authority_level="Hukuk Müşaviri",
        makam="Hukuk Müşavirliği Makamı",
        responsibilities=[
            "hukuki görüş bildirme", "dava ve icra takibi", "itirazların değerlendirilmesi",
            "sözleşme ve şartname hukuki incelemesi", "mevzuat uyum değerlendirmesi",
        ],
        handled_topics=["hukuki görüş", "dava", "itiraz", "sözleşme incelemesi", "mevzuat", "yargı"],
        keywords=["hukuki", "dava", "itiraz", "sözleşme", "mevzuat", "yargı", "icra"],
        entities=["Hukuk Müşavirliği"],
        legal_authority=["1136 sayılı Avukatlık Kanunu", "6100 sayılı Hukuk Muhakemeleri Kanunu"],
        required_documents=["dilekçe", "itiraz gerekçesi"],
        excluded_topics=["personel atama", "izin talebi", "satın alma"],
        routing_rules=["hukuki itiraz", "dava dilekçesi", "hukuki görüş talebi"],
        channel_hint="LEGAL",
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Mali Hizmetler Daire Başkanlığı",
        unit="Bütçe ve Ödeme Şube Müdürlüğü",
        authority_level="Daire Başkanı",
        makam="Mali Hizmetler Daire Başkanlığı Makamı",
        responsibilities=[
            "bütçe hazırlama ve uygulama", "ödeme ve tahakkuk işlemleri",
            "harcama belgelerinin kontrolü", "mali raporlama",
        ],
        handled_topics=["bütçe", "ödeme", "harcama", "mali", "tahakkuk", "fatura"],
        keywords=["bütçe", "ödeme", "harcama", "mali", "tahakkuk", "fatura", "avans"],
        entities=["Mali Hizmetler"],
        legal_authority=["5018 sayılı Kamu Mali Yönetimi ve Kontrol Kanunu"],
        required_documents=["harcama belgesi", "fatura"],
        excluded_topics=["personel disiplin", "hukuki itiraz"],
        routing_rules=["ödeme talebi", "bütçe talebi", "harcama onayı"],
        channel_hint=None,
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Bilgi İşlem Daire Başkanlığı",
        unit="Sistem Yönetimi Şube Müdürlüğü",
        authority_level="Daire Başkanı",
        makam="Bilgi İşlem Daire Başkanlığı Makamı",
        responsibilities=[
            "yazılım geliştirme ve bakım", "sistem arızalarının giderilmesi",
            "bilgi güvenliği önlemleri", "teknik destek hizmetleri",
        ],
        handled_topics=["yazılım", "sistem arızası", "bilgi güvenliği", "teknik destek", "ağ", "sunucu"],
        keywords=["yazılım", "sistem", "arıza", "bilgi güvenliği", "teknik destek", "sunucu", "ağ"],
        entities=["Bilgi İşlem"],
        legal_authority=["5651 sayılı Kanun"],
        required_documents=["arıza bildirim formu"],
        excluded_topics=["satın alma ihalesi", "personel atama"],
        routing_rules=["sistem arızası bildirimi", "teknik destek talebi"],
        channel_hint=None,
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Strateji Geliştirme Daire Başkanlığı",
        unit="Performans ve Kalite Şube Müdürlüğü",
        authority_level="Daire Başkanı",
        makam="Strateji Geliştirme Daire Başkanlığı Makamı",
        responsibilities=[
            "stratejik plan hazırlama", "performans programı izleme",
            "iç kontrol sistemi koordinasyonu",
        ],
        handled_topics=["stratejik plan", "performans programı", "iç kontrol", "izleme değerlendirme"],
        keywords=["stratejik plan", "performans", "iç kontrol", "izleme", "değerlendirme"],
        entities=["Strateji Geliştirme"],
        legal_authority=["5018 sayılı Kamu Mali Yönetimi ve Kontrol Kanunu"],
        required_documents=[],
        excluded_topics=["bireysel personel talebi"],
        routing_rules=["stratejik plan talebi", "performans raporu talebi"],
        channel_hint=None,
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Basın ve Halkla İlişkiler Müşavirliği",
        unit=None,
        authority_level="Müşavir",
        makam="Basın ve Halkla İlişkiler Müşavirliği Makamı",
        responsibilities=[
            "basın açıklamaları hazırlama", "medya ilişkileri yönetimi",
            "kamuoyu bilgilendirme faaliyetleri",
        ],
        handled_topics=["basın açıklaması", "medya", "kamuoyu", "sosyal medya"],
        keywords=["basın", "medya", "kamuoyu", "açıklama", "sosyal medya"],
        entities=["Basın ve Halkla İlişkiler"],
        legal_authority=[],
        required_documents=[],
        excluded_topics=["personel atama", "mali ödeme"],
        routing_rules=["basın açıklaması talebi", "medya talebi"],
        channel_hint=None,
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Özel Kalem Müdürlüğü",
        unit=None,
        authority_level="Özel Kalem Müdürü",
        makam="Başkanlık Özel Kalem Müdürlüğü Makamı",
        responsibilities=[
            "başkanlık yazışmalarının yönetimi", "protokol işlerinin takibi",
            "üst yazı ve randevu organizasyonu",
        ],
        handled_topics=["protokol", "başkanlık yazışmaları", "üst yazı", "randevu"],
        keywords=["protokol", "başkanlık", "üst yazı", "randevu", "makam"],
        entities=["Özel Kalem"],
        legal_authority=[],
        required_documents=[],
        excluded_topics=["teknik arıza", "mali ödeme"],
        routing_rules=["başkanlık makamına yazı", "protokol talebi"],
        channel_hint="CONFIDENTIAL",
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Teftiş Kurulu Başkanlığı",
        unit="İnceleme ve Soruşturma Şube Müdürlüğü",
        authority_level="Başkan",
        makam="Teftiş Kurulu Başkanlığı Makamı",
        responsibilities=[
            "inceleme ve soruşturma yürütme", "denetim faaliyetleri",
            "usulsüzlük iddialarının araştırılması",
        ],
        handled_topics=["inceleme", "soruşturma", "denetim", "usulsüzlük", "şikayet"],
        keywords=["inceleme", "soruşturma", "denetim", "usulsüzlük", "şikayet", "teftiş"],
        entities=["Teftiş Kurulu"],
        legal_authority=["7036 sayılı Kanun", "4483 sayılı Memurlar Hakkında Kanun"],
        required_documents=["şikayet dilekçesi"],
        excluded_topics=["rutin personel izni"],
        routing_rules=["soruşturma talebi", "şikayet dilekçesi", "usulsüzlük ihbarı"],
        channel_hint="CONFIDENTIAL",
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Destek Hizmetleri Daire Başkanlığı",
        unit="Satın Alma Şube Müdürlüğü",
        authority_level="Daire Başkanı",
        makam="Destek Hizmetleri Daire Başkanlığı Makamı",
        responsibilities=[
            "satın alma ve ihale süreçleri", "taşınır mal yönetimi", "lojistik hizmetler",
        ],
        handled_topics=["satın alma", "ihale", "taşınır", "lojistik", "demirbaş"],
        keywords=["satın alma", "ihale", "taşınır", "lojistik", "demirbaş", "tedarik"],
        entities=["Destek Hizmetleri"],
        legal_authority=["4734 sayılı Kamu İhale Kanunu"],
        required_documents=["talep formu", "teknik şartname"],
        excluded_topics=["personel izni", "hukuki dava"],
        routing_rules=["satın alma talebi", "ihale süreci başlatma"],
        channel_hint=None,
    ),
    Department(
        institution=_INSTITUTION,
        presidency=_PRESIDENCY,
        department="Dış İlişkiler Daire Başkanlığı",
        unit="Uluslararası İşbirliği Şube Müdürlüğü",
        authority_level="Daire Başkanı",
        makam="Dış İlişkiler Daire Başkanlığı Makamı",
        responsibilities=[
            "uluslararası işbirliği faaliyetleri", "yurt dışı görevlendirme koordinasyonu",
            "protokol anlaşmalarının takibi",
        ],
        handled_topics=["uluslararası", "yurt dışı", "protokol anlaşması", "işbirliği"],
        keywords=["uluslararası", "yurt dışı", "protokol anlaşması", "işbirliği", "heyet"],
        entities=["Dış İlişkiler"],
        legal_authority=[],
        required_documents=[],
        excluded_topics=["iç personel izni"],
        routing_rules=["uluslararası işbirliği talebi", "yurt dışı görevlendirme"],
        channel_hint="EXTERNAL_CORRESPONDENCE",
    ),
]


class KnowledgeBase:
    """Holds departments plus a pre-built BM25 index over their corpora."""

    def __init__(self, departments: Optional[List[Department]] = None):
        self.departments: List[Department] = departments if departments is not None else list(_SEED_DEPARTMENTS)
        self._index_map: Dict[str, int] = {d.id: i for i, d in enumerate(self.departments)}
        self.bm25 = BM25Index([d.corpus_text() for d in self.departments])

    def all(self) -> List[Department]:
        return list(self.departments)

    def index_map(self) -> Dict[str, int]:
        return self._index_map

    def get_by_name(self, department_name: str) -> Optional[Department]:
        for d in self.departments:
            if d.department == department_name:
                return d
        return None

    def search_by_topic(self, topic: str) -> List[Department]:
        topic_norm = topic.lower()
        return [d for d in self.departments if any(topic_norm in t.lower() for t in d.handled_topics)]


def default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
