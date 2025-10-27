import Navbar from "@/components/ui/custom/navbar";
import React from "react";

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <main className="flex min-h-screen">
      <Navbar />
      <div className="pt-20 w-full">{children}</div>
    </main>
  );
}
