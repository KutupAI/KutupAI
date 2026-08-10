import React from "react";
import KutupAIChat from "../components/KutupAIChat/KutupAIChat";

/**
 * صفحة المحادثة — تُربط بالـ Router الحالي في المشروع (مثال: React Router)
 * بنفس أسلوب باقي الصفحات (UploadPage.tsx, DashboardPage.tsx...).
 * لا يوجد منطق إضافي هنا عمداً؛ كل الحالة داخل KutupAIChat عبر useChat.
 */
const ChatPage: React.FC = () => <KutupAIChat />;

export default ChatPage;
