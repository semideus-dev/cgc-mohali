import { Fraunces, Space_Grotesk } from "next/font/google";

export const heading = Fraunces({
  variable: "--font-primary",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const primary = Space_Grotesk({
  variable: "--font-heading",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});
