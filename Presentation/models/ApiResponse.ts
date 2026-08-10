/**
 * يطابق شكل الاستجابة المستخدم فعلياً في المشروع (كما ورد في نمط http.post
 * الحالي): Success / Message / Code / AdditionalData / CarryOnData.
 * لا تُنشئ أي شكل استجابة جديد — كل خدمة جديدة يجب أن تُرجع/تتوقع هذا الشكل.
 */
export interface ApiResponse<TData = unknown> {
  Success: boolean;
  Message?: string;
  Code?: string | number;
  AdditionalData?: TData;
  CarryOnData?: unknown;
}
