"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { authClient } from "@/lib/auth-client";
import {
  Upload,
  Eye,
  Wand2,
  Target,
  MessageSquare,
  Palette,
  TrendingUp,
  BarChart3,
  Zap,
  CheckCircle,
  Star,
  Play,
  ArrowRight,
  Brain,
  Sparkles,

} from "lucide-react";


export default function LandingPage() {
  const { data: session } = authClient.useSession();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      

      {/* 🏠 1. Hero Section — The Wow Moment */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-white to-primary/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
          <div className="text-center">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-8">
              <Brain className="w-4 h-4 mr-2" />
              AI-Powered Creative Intelligence
            </div>
            <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6">
              Transform Your Ads with
              <span className="block text-primary">AI-Powered Creative</span>
              Intelligence
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-4xl mx-auto">
              Upload your ad - get expert AI feedback, visual insights, and better-performing
              alternatives in seconds. No design experience needed.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <Button size="lg" className="px-8 py-4 text-lg bg-primary hover:bg-primary/90">
                <Zap className="w-5 h-5 mr-2" />
                Try It Now
              </Button>
              <Button size="lg" variant="outline" className="px-8 py-4 text-lg">
                <Play className="w-5 h-5 mr-2" />
                See Demo
              </Button>
            </div>


          </div>
        </div>

        {/* Background decoration */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-primary/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
        </div>
      </section>

      {/* 🤖 2. How It Works (3-step visual flow) */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Transform your ads in three simple steps with our AI-powered platform
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Step 1 */}
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow relative">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <Badge className="bg-primary text-white px-3 py-1">Step 1</Badge>
                </div>
                <h3 className="text-xl font-semibold mb-4">Upload Your Ad</h3>
                <p className="text-gray-600 mb-4">
                  Upload your ad – JPG, PNG, or banner format. Our AI instantly begins analysis.
                </p>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                    <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-500">Drag & drop your ad here</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 2 */}
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow relative">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/15 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Eye className="w-8 h-8 text-primary" />
                </div>
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <Badge className="bg-primary text-white px-3 py-1">Step 2</Badge>
                </div>
                <h3 className="text-xl font-semibold mb-4">AI Analyzes Everything</h3>
                <p className="text-gray-600 mb-4">
                  AI analyzes visuals and text – detects composition, contrast, and emotional tone.
                </p>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <Brain className="w-6 h-6 text-primary" />
                    <BarChart3 className="w-6 h-6 text-primary/70" />
                  </div>
                  <div className="space-y-2">
                    <div className="bg-primary/30 h-2 rounded-full w-3/4 mx-auto"></div>
                    <div className="bg-primary/20 h-2 rounded-full w-1/2 mx-auto"></div>
                    <div className="bg-primary/40 h-2 rounded-full w-2/3 mx-auto"></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 3 */}
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow relative">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Wand2 className="w-8 h-8 text-primary" />
                </div>
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <Badge className="bg-primary text-white px-3 py-1">Step 3</Badge>
                </div>
                <h3 className="text-xl font-semibold mb-4">Get Better Versions</h3>
                <p className="text-gray-600 mb-4">
                  Get better ad versions instantly – download or compare variations with improved performance.
                </p>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-gradient-to-br from-primary to-primary/70 h-12 rounded"></div>
                    <div className="bg-gradient-to-br from-primary/80 to-primary h-12 rounded"></div>
                    <div className="bg-gradient-to-br from-primary/60 to-primary/90 h-12 rounded"></div>
                    <div className="bg-gradient-to-br from-primary/90 to-primary/60 h-12 rounded"></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Connection arrows */}
            <div className="hidden md:block absolute top-1/2 left-1/3 transform -translate-y-1/2 -translate-x-1/2">
              <ArrowRight className="w-6 h-6 text-gray-400" />
            </div>
            <div className="hidden md:block absolute top-1/2 right-1/3 transform -translate-y-1/2 translate-x-1/2">
              <ArrowRight className="w-6 h-6 text-gray-400" />
            </div>
          </div>
        </div>
      </section>

      {/* 📊 3. Live Example / Demo Preview */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              See the Transformation
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Watch how our AI transforms ordinary ads into high-performing creatives
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            <Card className="border-0 shadow-2xl overflow-hidden">
              <CardContent className="p-0">
                <div className="grid grid-cols-1 md:grid-cols-2">
                  {/* Before */}
                  <div className="p-8 bg-gray-50">
                    <div className="text-center mb-4">
                      <Badge variant="destructive" className="mb-2">Before</Badge>
                      <h3 className="font-semibold text-gray-900">Original Ad</h3>
                    </div>
                    <div className="bg-white rounded-lg p-6 shadow-inner">
                      <div className="bg-gray-200 h-32 rounded mb-4 flex items-center justify-center">
                        <span className="text-gray-500 text-sm">Low contrast, poor layout</span>
                      </div>
                      <div className="space-y-2">
                        <div className="bg-gray-300 h-3 rounded w-3/4"></div>
                        <div className="bg-gray-300 h-3 rounded w-1/2"></div>
                      </div>
                    </div>
                    <div className="mt-4 space-y-2 text-sm">
                      <div className="flex items-center text-red-600">
                        <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                        Visual Balance: 45/100
                      </div>
                      <div className="flex items-center text-red-600">
                        <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                        Text Readability: Poor
                      </div>
                      <div className="flex items-center text-red-600">
                        <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                        Emotion Score: 32% Positive
                      </div>
                    </div>
                  </div>

                  {/* After */}
                  <div className="p-8 bg-primary/5">
                    <div className="text-center mb-4">
                      <Badge className="mb-2 bg-primary">After AI Enhancement</Badge>
                      <h3 className="font-semibold text-gray-900">Optimized Ad</h3>
                    </div>
                    <div className="bg-white rounded-lg p-6 shadow-inner">
                      <div className="bg-gradient-to-br from-primary to-primary/80 h-32 rounded mb-4 flex items-center justify-center">
                        <span className="text-white text-sm font-medium">High contrast, perfect layout</span>
                      </div>
                      <div className="space-y-2">
                        <div className="bg-gray-800 h-3 rounded w-3/4"></div>
                        <div className="bg-gray-600 h-3 rounded w-1/2"></div>
                      </div>
                    </div>
                    <div className="mt-4 space-y-2 text-sm">
                      <div className="flex items-center text-green-600">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Visual Balance: 92/100
                      </div>
                      <div className="flex items-center text-green-600">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Text Readability: Excellent
                      </div>
                      <div className="flex items-center text-green-600">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Emotion Score: 87% Positive
                      </div>
                    </div>
                  </div>
                </div>

                {/* Improvement banner */}
                <div className="bg-primary text-white p-4 text-center">
                  <p className="font-semibold">
                    <TrendingUp className="w-4 h-4 inline mr-2" />
                    Boosted contrast & stronger CTA — 156% better performance predicted
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* 💡 4. AI Insights Dashboard Preview */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              AI Insights Dashboard
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Get detailed analytics and actionable insights for every ad
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900">Visual Balance</h3>
                  <BarChart3 className="w-5 h-5 text-primary" />
                </div>
                <div className="text-3xl font-bold text-primary mb-2">82/100</div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-primary h-2 rounded-full" style={{ width: '82%' }}></div>
                </div>
                <p className="text-sm text-gray-600 mt-2">Good composition balance</p>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900">Text Readability</h3>
                  <Eye className="w-5 h-5 text-primary" />
                </div>
                <div className="text-3xl font-bold text-primary mb-2">Excellent</div>
                <div className="flex space-x-1 mb-2">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-sm text-gray-600">Perfect contrast ratio</p>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900">Emotion Score</h3>
                  <Brain className="w-5 h-5 text-primary" />
                </div>
                <div className="text-3xl font-bold text-primary mb-2">74%</div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-primary h-2 rounded-full" style={{ width: '74%' }}></div>
                </div>
                <p className="text-sm text-gray-600 mt-2">Positive emotional impact</p>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900">ROI Prediction</h3>
                  <TrendingUp className="w-5 h-5 text-primary" />
                </div>
                <div className="text-3xl font-bold text-primary mb-2">+156%</div>
                <div className="flex items-center space-x-1 mb-2">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-green-600 font-medium">High potential</span>
                </div>
                <p className="text-sm text-gray-600">Expected performance boost</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* 🧠 5. Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Powerful AI Features
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Everything you need to create high-performing ads with AI assistance
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Target className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Smart Visual Analysis</h3>
                <p className="text-gray-600">
                  AI analyzes composition, color harmony, and visual hierarchy to optimize your ad's impact.
                </p>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/15 rounded-full flex items-center justify-center mx-auto mb-6">
                  <MessageSquare className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Copywriting Feedback</h3>
                <p className="text-gray-600">
                  Get instant feedback on your ad copy with suggestions for better engagement and conversion.
                </p>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Palette className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">AI Ad Redesigns</h3>
                <p className="text-gray-600">
                  Generate multiple design variations automatically with improved layouts and visual appeal.
                </p>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-primary/25 rounded-full flex items-center justify-center mx-auto mb-6">
                  <TrendingUp className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">ROI-Driven Recommendations</h3>
                <p className="text-gray-600">
                  Receive data-backed suggestions to maximize your ad performance and return on investment.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl sm:text-4xl font-bold mb-6">
            Ready to Transform Your Ads?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Join thousands of marketers who are already using AdVision to create
            better-performing ads with AI.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" variant="secondary" className="px-8 py-4 text-lg">
              <Sparkles className="w-5 h-5 mr-2" />
              Start Free Trial
            </Button>
            <Button size="lg" variant="outline" className="px-8 py-4 text-lg border-white text-white hover:bg-white hover:text-primary">
              Contact Sales
            </Button>
          </div>
          <div className="mt-6 text-sm opacity-75">
            <CheckCircle className="w-4 h-4 inline mr-2" />
            No credit card required • 7-day free trial • Cancel anytime
          </div>
        </div>
      </section>
    </div>
  );
}