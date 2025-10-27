"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import {
    Mail,
    Crown,
    Zap,
    Star,
    Check,
    CreditCard,
    Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface CreditPlan {
    id: string;
    name: string;
    price: number;
    credits: number;
    features: string[];
    popular?: boolean;
    badge?: string;
}

interface UserCredits {
    current: number;
    total: number;
    plan: string;
    resetDate: string;
}

export default function ProfilePage() {
    const { data: session, isPending: isLoading } = authClient.useSession();
    const [showUpgradeModal, setShowUpgradeModal] = useState(false);


    const [userCredits] = useState<UserCredits>({
        current: 25,
        total: 100,
        plan: "Free",
        resetDate: "2024-02-01"
    });

    const creditPlans: CreditPlan[] = [
        {
            id: "free",
            name: "Free",
            price: 0,
            credits: 100,
            features: [
                "100 AI Canvas generations",
                "Basic templates",
                "Community support",
                "Export in PNG/JPG",
                "Limited analytics"
            ]
        },
        {
            id: "pro",
            name: "Pro",
            price: 29,
            credits: 2000,
            popular: true,
            badge: "Most Popular",
            features: [
                "2,000 AI Canvas generations",
                "Premium templates",
                "Priority support",
                "Export in all formats",
                "Custom branding",
                "Advanced analytics",
                "API access"
            ]
        },
        {
            id: "enterprise",
            name: "Enterprise",
            price: 99,
            credits: 10000,
            badge: "Best Value",
            features: [
                "10,000 AI Canvas generations",
                "Unlimited templates",
                "24/7 dedicated support",
                "White-label solution",
                "Full API access",
                "Team collaboration",
                "Advanced analytics",
                "Custom integrations"
            ]
        }
    ];

    if (isLoading) {
        return (
            <div className="container mx-auto py-8 px-4">
                <div className="max-w-4xl mx-auto space-y-6">
                    <div className="flex items-center space-x-4">
                        <Skeleton className="h-20 w-20 rounded-full" />
                        <div className="space-y-2">
                            <Skeleton className="h-8 w-48" />
                            <Skeleton className="h-4 w-64" />
                        </div>
                    </div>
                    <Skeleton className="h-64 w-full" />
                </div>
            </div>
        );
    }

    if (!session?.user) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
                <div className="w-full max-w-md text-center space-y-6">
                    <h1 className="text-xl sm:text-2xl md:text-3xl font-bold">Not Signed In</h1>
                    <p className="text-sm sm:text-base text-muted-foreground px-4">
                        Please sign in to view your profile.
                    </p>
                    <Button asChild className="w-full sm:w-auto px-8">
                        <a href="/sign-in">Sign In</a>
                    </Button>
                </div>
            </div>
        );
    }

    const { user } = session;
    const initials = user.name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase();

    return (
        <div className="flex flex-col items-center pt-4 sm:pt-8 w-full min-h-screen bg-background px-4 sm:px-0">
            <div className="w-full sm:w-[90%] md:w-[80%] max-w-6xl flex flex-col gap-4 sm:gap-8">
                {/* Profile Header */}
                <div className="border border-b-2 border-b-primary p-4 rounded-xl flex items-center justify-between bg-white">
                    <div className="flex items-center space-x-2 md:space-x-6">
                        <Avatar className="h-10 w-10 md:h-20 md:w-20">
                            <AvatarFallback className="bg-primary text-primary-foreground text-lg md:text-2xl font-semibold">
                                {initials}
                            </AvatarFallback>
                        </Avatar>
                        <div className="space-y-1">
                            <h1 className="text-xl md:text-3xl font-bold">{user.name}</h1>
                            <p className="text-xs md:text-lg text-muted-foreground flex items-center gap-2">
                                <Mail className="h-4 w-4" />
                                {user.email}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                            variant="destructive"
                            onClick={() => authClient.signOut()}
                            className="text-xs sm:text-sm md:text-base px-2 sm:px-3 md:px-4 py-1 sm:py-2"
                        >
                            Sign Out
                        </Button>
                    </div>
                </div>

                {/* Low Credits Warning - Floating */}
                {userCredits.current < 10 && (
                    <div className="fixed bottom-4 right-4 left-4 sm:left-auto sm:bottom-6 sm:right-6 z-50 animate-bounce">
                        <Card className="bg-gradient-to-r from-orange-500 to-red-500 text-white border-0 shadow-lg">
                            <CardContent className="p-3 sm:p-4">
                                <div className="flex items-center gap-2 sm:gap-3">
                                    <div className="bg-white/20 rounded-full p-1.5 sm:p-2 flex-shrink-0">
                                        <Zap className="h-4 w-4 sm:h-5 sm:w-5" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="font-semibold text-xs sm:text-sm text-orange-200">Credits Running Low!</p>
                                        <p className="text-xs opacity-90">Only {userCredits.current} credits left</p>
                                    </div>
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        className="bg-white text-orange-600 hover:bg-gray-100 text-xs px-2 sm:px-3 flex-shrink-0"
                                        onClick={() => setShowUpgradeModal(true)}
                                    >
                                        <Star className="h-3 w-3 mr-1" />
                                        <span className="hidden sm:inline">Upgrade</span>
                                        <span className="sm:hidden">Up</span>
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Profile Overview */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-lg sm:text-xl">Profile Overview</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div>
                                <h3 className="font-semibold mb-3 text-sm sm:text-base">Account Information</h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Full Name:</span>
                                        <span className="break-words">{user.name}</span>
                                    </div>
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Email:</span>
                                        <span className="break-all">{user.email}</span>
                                    </div>
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Plan:</span>
                                        <Badge variant="secondary" className="w-fit">{userCredits.plan}</Badge>
                                    </div>
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Member Since:</span>
                                        <span>January 2024</span>
                                    </div>
                                </div>
                            </div>

                            <div>
                                <h3 className="font-semibold mb-3 text-sm sm:text-base">Account Activity</h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Last Login:</span>
                                        <span>Today</span>
                                    </div>
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Total Canvas:</span>
                                        <span>47</span>
                                    </div>
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Projects:</span>
                                        <span>12</span>
                                    </div>
                                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 sm:gap-2">
                                        <span className="text-muted-foreground font-medium">Status:</span>
                                        <Badge variant="success" className="w-fit">Active</Badge>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Credits Section */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Zap className="h-5 w-5 text-yellow-500" />
                            Credits & Billing
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="flex flex-col gap-6">
                            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                                <div className="flex-1 w-full">
                                    <div className="flex items-center gap-3 mb-2">
                                        <span className="text-xl sm:text-2xl font-bold">{userCredits.current}</span>
                                        <span className="text-sm sm:text-base text-muted-foreground">/ {userCredits.total} credits</span>
                                    </div>
                                    <div className="w-full max-w-sm bg-gray-200 rounded-full h-2 mb-2">
                                        <div
                                            className="bg-primary h-2 rounded-full transition-all duration-300"
                                            style={{ width: `${(userCredits.current / userCredits.total) * 100}%` }}
                                        />
                                    </div>
                                    <p className="text-xs sm:text-sm text-muted-foreground">
                                        Credits reset on {new Date(userCredits.resetDate).toLocaleDateString()}
                                    </p>
                                </div>

                                <Dialog open={showUpgradeModal} onOpenChange={setShowUpgradeModal}>
                                    <DialogTrigger asChild>
                                        <Button variant="outline" className="w-full sm:w-auto px-6">
                                            <Crown className="h-4 w-4 mr-2" />
                                            Upgrade Plan
                                        </Button>
                                    </DialogTrigger>
                                    <DialogContent className="max-w-[95vw] sm:max-w-4xl max-h-[90vh] overflow-y-auto">
                                        <DialogHeader>
                                            <DialogTitle className="text-2xl font-bold text-center">
                                                <span className="text-primary">
                                                    Upgrade Your Plan
                                                </span>
                                            </DialogTitle>
                                            <DialogDescription className="text-center">
                                                Choose the perfect plan for your canvas creation needs
                                            </DialogDescription>
                                        </DialogHeader>

                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 py-6">
                                            {creditPlans.map((plan) => (
                                                <div
                                                    key={plan.id}
                                                    className={cn(
                                                        "relative rounded-xl border-2 p-6 transition-all duration-200 hover:shadow-lg",
                                                        plan.popular
                                                            ? "border-primary bg-primary/5 scale-105"
                                                            : "border-gray-200 hover:border-gray-300"
                                                    )}
                                                >
                                                    {plan.badge && (
                                                        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                                                            <Badge
                                                                variant={plan.popular ? "default" : "secondary"}
                                                                className={cn(
                                                                    "px-3 py-1",
                                                                    plan.popular && "bg-primary text-primary-foreground"
                                                                )}
                                                            >
                                                                {plan.badge}
                                                            </Badge>
                                                        </div>
                                                    )}

                                                    <div className="text-center space-y-4">
                                                        <div>
                                                            <h3 className="text-xl font-bold">{plan.name}</h3>
                                                            <div className="mt-2">
                                                                {plan.price === 0 ? (
                                                                    <span className="text-3xl font-bold text-green-600">Free</span>
                                                                ) : (
                                                                    <>
                                                                        <span className="text-3xl font-bold">${plan.price}</span>
                                                                        <span className="text-muted-foreground">/month</span>
                                                                    </>
                                                                )}
                                                            </div>
                                                        </div>

                                                        <div className="flex items-center justify-center gap-2 text-sm">
                                                            <Sparkles className="h-4 w-4 text-yellow-500" />
                                                            <span className="font-medium">{plan.credits.toLocaleString()} Credits</span>
                                                        </div>

                                                        <ul className="space-y-2 text-sm">
                                                            {plan.features.map((feature, index) => (
                                                                <li key={index} className="flex items-center gap-2">
                                                                    <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
                                                                    <span>{feature}</span>
                                                                </li>
                                                            ))}
                                                        </ul>

                                                        <Button
                                                            className="w-full"
                                                            variant={plan.popular ? "default" : "outline"}
                                                            disabled={plan.price === 0 && userCredits.plan === "Free"}
                                                        >
                                                            {plan.price === 0 ? (
                                                                userCredits.plan === "Free" ? (
                                                                    <>
                                                                        <Check className="h-4 w-4 mr-2" />
                                                                        Current Plan
                                                                    </>
                                                                ) : (
                                                                    <>
                                                                        <Check className="h-4 w-4 mr-2" />
                                                                        Choose Free
                                                                    </>
                                                                )
                                                            ) : (
                                                                <>
                                                                    <CreditCard className="h-4 w-4 mr-2" />
                                                                    Choose {plan.name}
                                                                </>
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>

                                        <DialogFooter className="flex-col sm:flex-row gap-2">
                                            <p className="text-xs text-muted-foreground text-center">
                                                All plans include a 7-day free trial. Cancel anytime.
                                            </p>
                                        </DialogFooter>
                                    </DialogContent>
                                </Dialog>
                            </div>

                            <div className="text-sm text-muted-foreground">
                                <p>• Each AI Canvas generation costs 1 credit</p>
                                <p>• Template customization costs 1 credit</p>
                                <p>• Unused credits roll over to next month</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Quick Actions */}
                <Card>
                    <CardHeader>
                        <CardTitle>Quick Actions</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                            <Button variant="outline" className="h-20 flex-col gap-2">
                                <Sparkles className="h-6 w-6" />
                                <span>Create Canvas</span>
                            </Button>
                            <Button variant="outline" className="h-20 flex-col gap-2">
                                <Sparkles className="h-6 w-6" />
                                <span>View Analytics</span>
                            </Button>
                            <Button variant="outline" className="h-20 flex-col gap-2">
                                <Crown className="h-6 w-6" />
                                <span>Upgrade Plan</span>
                            </Button>
                            <Button variant="outline" className="h-20 flex-col gap-2">
                                <CreditCard className="h-6 w-6" />
                                <span>Buy Credits</span>
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}