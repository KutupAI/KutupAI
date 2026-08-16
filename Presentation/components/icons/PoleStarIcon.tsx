import React from "react";

interface PoleStarIconProps {
  size?: number;
  className?: string;
  animated?: boolean;
}

/**
 * أيقونة "Kutup Yıldızı" (نجمة القطب) — العنصر البصري التوقيعي لـ KutupAI.
 * مستوحاة مباشرة من اسم المنتج (Kutup = قطب/شمالي) بدل أي رمز سياحي عام،
 * وتُستخدم في: الشعار، شاشة الترحيب، وحالة "يفكّر" أثناء انتظار الرد.
 */
const PoleStarIcon: React.FC<PoleStarIconProps> = ({
  size = 24,
  className = "",
  animated = false,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 32 32"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    style={animated ? { animation: "kutup-star-pulse 1.8s ease-in-out infinite" } : undefined}
    aria-hidden="true"
  >
    <path
      d="M16 2 L18.4 13.6 L30 16 L18.4 18.4 L16 30 L13.6 18.4 L2 16 L13.6 13.6 Z"
      fill="url(#kutupStarGradient)"
    />
    <circle cx="16" cy="16" r="2.4" fill="var(--kutup-white, #F6F6F8)" />
    <defs>
      <linearGradient id="kutupStarGradient" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="var(--kutup-red-hover, #F0453D)" />
        <stop offset="100%" stopColor="var(--kutup-red, #E0332C)" />
      </linearGradient>
    </defs>
  </svg>
);

export default PoleStarIcon;
