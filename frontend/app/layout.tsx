import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ModelForge · ML Workspace",
  description: "Train, compare, and monitor reproducible models.",
};

export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
