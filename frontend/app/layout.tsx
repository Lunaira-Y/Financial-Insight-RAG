import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial Insight RAG | Professional Edition",
  description: "High-end financial analysis powered by AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
