import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#050511",
        surface: "rgba(255, 255, 255, 0.03)",
        primary: {
          DEFAULT: "#6366f1", // Indigo
          glow: "rgba(99, 102, 241, 0.5)",
        },
        secondary: {
          DEFAULT: "#a855f7", // Purple
          glow: "rgba(168, 85, 247, 0.5)",
        },
        accent: "#22d3ee", // Cyan
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "primary-gradient": "linear-gradient(90deg, #6366f1 0%, #a855f7 100%)",
        "card-gradient": "linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%)",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;





