# اختبار طبقة Application

الدليل الكامل للمدخلات/المخرجات والشروط: **`Application/README.md`**.

## اختبار سريع للعقد (Orchestration)

```powershell
cd D:\AI\KutupAI
python -m pytest Orchestration/tests/test_state_manager.py::test_initialize_from_application_envelope Orchestration/tests/test_process_service.py::test_run_workflow_from_application_envelope -v
```

## تجربة يدوية Application ↔ Orchestration

1. شغّل Orchestration على `:8000` و Application على `:8080` (انظر `Application/README.md`).
2. أرسل ملفًا واحدًا + سؤال عبر `POST /api/Chat/SendMessage`.
3. راقب `{ Success, Data }`؛ للملفات >10MB أو الأنواع غير المسموحة أو `File` كمصفوفة تتوقع HTTP 400 من Application قبل الوصول لـ Orchestration.
