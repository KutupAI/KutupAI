// General formatting helpers, no React dependency.

/**
 * Ham kaynak dosya adını okunabilir kanun başlığına çevirir.
 * Örnek: "1076_Yedek Subaylar Kanunu.pdf" -> "Yedek Subaylar Kanunu"
 * Gerçek veri setinden doğrulanmış desen: "<kanun_no>_<Başlık>.pdf".
 * Desen eşleşmezse (örn. farklı bir dosya adlandırması), dosya adı .pdf
 * uzantısı çıkarılmış haliyle olduğu gibi döner -- asla boş göstermez.
 */
export const formatLawTitle = (sourceFile: string | null): string => {
  if (!sourceFile) return "Bilinmeyen kaynak";
  const withoutExt = sourceFile.replace(/\.pdf$/i, "");
  const match = withoutExt.match(/^\d+_(.+)$/);
  return (match ? match[1] : withoutExt).trim() || "Bilinmeyen kaynak";
};

/** "Madde 3" / null -- article_no bazen "unknown" string'i olarak geliyor
 *  (facts_registry.json'da görüldü), bunu da null gibi ele alır. */
export const formatArticleLabel = (articleNo: string | null): string | null => {
  if (!articleNo || articleNo.toLowerCase() === "unknown") return null;
  return `Madde ${articleNo}`;
};

/** "s. 12" ya da "s. 12–14" (page_start/page_end'den). */
export const formatPageRange = (start: number | null, end: number | null): string | null => {
  if (start === null) return null;
  return end !== null && end !== start ? `s. ${start}–${end}` : `s. ${start}`;
};

/** "Bugün" / "Dün" / "Bu hafta" / "Daha eski" gruplamaları -- Claude/ChatGPT
 *  tarzı sidebar geçmişi için. `updatedAt` epoch ms bekler. */
export const groupChatsByRecency = <T extends { updatedAt: number }>(
  items: T[]
): { label: string; items: T[] }[] => {
  const now = new Date();
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const today = startOfDay(now);
  const yesterday = today - 86_400_000;
  const weekAgo = today - 7 * 86_400_000;

  const buckets = { today: [] as T[], yesterday: [] as T[], week: [] as T[], older: [] as T[] };

  for (const item of items) {
    if (item.updatedAt >= today) buckets.today.push(item);
    else if (item.updatedAt >= yesterday) buckets.yesterday.push(item);
    else if (item.updatedAt >= weekAgo) buckets.week.push(item);
    else buckets.older.push(item);
  }

  return [
    { label: "Bugün", items: buckets.today },
    { label: "Dün", items: buckets.yesterday },
    { label: "Bu hafta", items: buckets.week },
    { label: "Daha eski", items: buckets.older },
  ].filter((g) => g.items.length > 0);
};
