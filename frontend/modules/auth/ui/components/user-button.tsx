"use client";

import { authClient } from "@/lib/auth-client";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function UserButton() {
  const { data: session, isPending: isLoading } = authClient.useSession();

  if (isLoading) {
    return <Skeleton className="h-10 w-10 rounded-full" />;
  }

  return session == null ? (
    <Link href="/sign-in">
      <Button variant="link" effect="underline">
        Sign In
      </Button>
    </Link>
  ) : (
    <Link href="/profile" className="p-0 m-0">
      <Avatar>
        <AvatarFallback className="bg-white text-black">
          {session.user.name
            .split(" ")
            .map((n) => n[0])
            .join("")
            .toUpperCase()}
        </AvatarFallback>
      </Avatar>
    </Link>
  );
}
