"use client";

import { Button } from "@/components/ui/button";
import { authClient } from "@/lib/auth-client";

export default function LandingPage() {
  return (
    <div className="flex h-screen items-center flex-col gap-4 justify-center">
      <h1 className="text-4xl font-semibold">AdVision</h1>
      <div className="flex gap-4">
        <Button asChild className="px-6 py-2">
          <a href="/profile">View Profile</a>
        </Button>
        <Button
          variant="destructive"
          onClick={() => authClient.signOut()}
          className="text-xs sm:text-sm md:text-base px-2 sm:px-3 md:px-4 py-1 sm:py-2"
        >
          Sign Out
        </Button>
      </div>
    </div>
  );
}
