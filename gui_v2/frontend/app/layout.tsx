import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Mike Agent | Zyrix UI",
  description: "AI-Powered Trading Agent Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="fixed inset-0 z-[-1] pointer-events-none">
           <div className="absolute top-0 left-0 w-[500px] h-[500px] bg-primary/10 rounded-full blur-[128px] -translate-x-1/2 -translate-y-1/2" />
           <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-secondary/10 rounded-full blur-[128px] translate-x-1/2 translate-y-1/2" />
        </div>
        {children}
      </body>
    </html>
  );
}





