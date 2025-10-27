"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import SignUpForm from "@/modules/auth/ui/components/sign-up-form";
import Link from "next/link";
import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { heading } from "@/components/providers/font-provider";

export default function SignUpView() {
  const { data: session, isPending: isLoading } = authClient.useSession();
  const router = useRouter();

  useEffect(() => {
    
    if (session?.user && !isLoading) {
      router.push("/");
    }
  }, [session, isLoading, router]);

 
  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4 sm:p-6 md:p-8">
        <Card className="bg-background w-[90%] md:w-[40%] border-b-2 border-b-primary">
          <CardHeader>
            <Skeleton className="h-12 w-32" />
            <Skeleton className="h-4 w-48" />
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // If user is signed in, don't render the form (redirect will happen)
  if (session?.user) {
    return null;
  }

  return (
    <div className="flex min-h-screen w-full items-center justify-center p-4 sm:p-6 md:p-8">
      <Card className="bg-background w-[90%] md:w-[40%] border-b-2 border-b-primary">
        <CardHeader>
          <CardTitle
            className={cn(
              "text-4xl font-semibold tracking-tight text-primary"
            )}
          >
            <Link href="/" className={cn(heading.className, "hover:opacity-80 transition-opacity cursor-pointer")}>
              AdVision
            </Link>
          </CardTitle>
          <CardDescription>
            Already have an account?{" "}
            <Link
              href="/sign-in"
              className="text-primary underline underline-offset-4 hover:underline-offset-8 duration-200"
            >
              Sign In
            </Link>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SignUpForm />
        </CardContent>
      </Card>
    </div>
  );
}
