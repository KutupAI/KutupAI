import React from "react";

interface PoleStarIconProps {
  size?: number;
  className?: string;
  animated?: boolean;
  /** Adds the cool "starlight" halo behind the star — reserved for the
   *  welcome/hero moment so it stays a signature beat, not background noise. */
  glow?: boolean;
}


 
const PoleStarIcon: React.FC<PoleStarIconProps> = ({
  size = 24,
  className = "",
  animated = false,
  glow = false,
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
    {glow && (
      <circle cx="16" cy="16" r="15" fill="url(#kutupStarHalo)" style={{ animation: "kutup-starlight-breathe 4.5s ease-in-out infinite" }} />
    )}

    <path
      d="M16 6.5 L17.1 14.9 L25.5 16 L17.1 17.1 L16 25.5 L14.9 17.1 L6.5 16 L14.9 14.9 Z"
      fill="var(--kutup-starlight, #CFE0FF)"
      opacity="0.55"
    />

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
      <radialGradient id="kutupStarHalo" cx="0.5" cy="0.5" r="0.5">
        <stop offset="0%" stopColor="var(--kutup-starlight, #CFE0FF)" stopOpacity="0.5" />
        <stop offset="100%" stopColor="var(--kutup-starlight, #CFE0FF)" stopOpacity="0" />
      </radialGradient>
    </defs>
  </svg>
);

export default PoleStarIcon;
