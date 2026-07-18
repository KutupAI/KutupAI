from dataclasses import dataclass
from typing import List
# الملف عبارة عن كلاس بيانات فقط (DTO) عقد يعني بس 
#كلاس من نوع داتا يحتوي عتنين بارامتر الاول لليوزر والثاني للسؤال من نوع سترينغ 
@dataclass(slots=True)
class Message:
    role: str
    content: str

# وهون التنفيذ تبعه 
@dataclass(slots=True)
class InferenceRequest:
    messages: List[Message]#للرسالة 

    temperature: float = 0.2#لتحديد درجة العشوائية في الاجابة
    top_p: float = 0.9#لتحديد احتمال الاختيار   
    max_tokens: int = 512#لتحديد الحد الأقصى لعدد الرموز في الاجابة
    stream: bool = False#لتحديد ما إذا كان سيتم تدفق الإجابة بشكل مباشر او شوي شوي 
    