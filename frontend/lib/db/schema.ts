import { boolean, pgTable, text, timestamp, uuid, jsonb, varchar, index } from "drizzle-orm/pg-core";

// User authentication and management
export const user = pgTable("user", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  email: text("email").notNull().unique(),
  emailVerified: boolean("email_verified")
    .$defaultFn(() => false)
    .notNull(),
  image: text("image"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
}, (table) => ({
  emailIdx: index("ix_user_email").on(table.email),
}));

// User sessions for authentication
export const session = pgTable("session", {
  id: text("id").primaryKey(),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  token: text("token").notNull().unique(),
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  ipAddress: text("ip_address"),
  userAgent: text("user_agent"),
  userId: text("user_id")
    .notNull()
    .references(() => user.id, { onDelete: "cascade" }),
}, (table) => ({
  tokenIdx: index("ix_session_token").on(table.token),
}));

// OAuth accounts
export const account = pgTable("account", {
  id: text("id").primaryKey(),
  accountId: text("account_id").notNull(),
  providerId: text("provider_id").notNull(),
  userId: text("user_id")
    .notNull()
    .references(() => user.id, { onDelete: "cascade" }),
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  idToken: text("id_token"),
  accessTokenExpiresAt: timestamp("access_token_expires_at", { withTimezone: true }),
  refreshTokenExpiresAt: timestamp("refresh_token_expires_at", { withTimezone: true }),
  scope: text("scope"),
  password: text("password"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
});

// Email/phone verification
export const verification = pgTable("verification", {
  id: text("id").primaryKey(),
  identifier: text("identifier").notNull(),
  value: text("value").notNull(),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date()),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date()),
});

// Canvas/design storage
export const canvas = pgTable("canvas", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  url: text("url"),
  prompt: text("prompt").notNull(),
  userId: text("user_id")
    .notNull()
    .references(() => user.id, { onDelete: "cascade" }),
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
});

// Comprehensive analysis jobs table
export const analysisJobs = pgTable("analysis_jobs", {
  // Primary identification
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id").references(() => user.id, { onDelete: "cascade" }),
  status: varchar("status", { length: 50 }).$default(() => "pending").notNull(),
  
  // Timestamps
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  
  // Input data
  originalImageUrl: varchar("original_image_url", { length: 500 }),
  userPrompt: text("user_prompt"),
  
  // Analysis results (stored as JSONB for flexibility)
  ocrResults: jsonb("ocr_results").$type<{
    raw_text: string;
    segments: string[];
    confidence: number;
    word_count: number;
    char_count: number;
    segment_count: number;
    error?: string;
  }>(),
  
  textAnalysis: jsonb("text_analysis").$type<{
    sentiment: {
      label: string;
      type: string;
      score: number;
      confidence: string;
    };
    emotions: Record<string, number>;
    keywords: string[];
    readability_score: number;
    tone: string;
    language: string;
    entities: Array<{ type: string; value: string }>;
    text_quality: {
      length: string;
      has_call_to_action: boolean;
      has_brand_name: boolean;
      uses_numbers: boolean;
      punctuation_variety: number;
    };
    call_to_action: boolean;
    error?: string;
  }>(),
  
  imageAnalysis: jsonb("image_analysis").$type<{
    objects: {
      objects: Record<string, { count: number; max_confidence: number }>;
      total_detections: number;
    };
    colors: {
      dominant_colors: Array<{ hex: string; percentage: number }>;
      primary_color: string;
    };
    composition?: Record<string, any>;
    quality_metrics?: Record<string, number>;
    visual_hierarchy?: string[];
    error?: string;
  }>(),
  
  // Critique and generation
  critique: jsonb("critique").$type<{
    overall_score: number;
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
    target_audience: string;
    emotional_impact: string;
    effectiveness_rating: string;
    error?: string;
  }>(),
  
  masterPrompt: text("master_prompt"),
  generatedImageUrl: varchar("generated_image_url", { length: 500 }),
  
  // Legacy field for backward compatibility
  results: jsonb("results").$type<Record<string, any>>(),
}, (table) => ({
  idIdx: index("ix_analysis_jobs_id").on(table.id),
  statusIdx: index("ix_analysis_jobs_status").on(table.status),
}));

// Type exports for TypeScript usage
export type User = typeof user.$inferSelect;
export type NewUser = typeof user.$inferInsert;

export type Session = typeof session.$inferSelect;
export type NewSession = typeof session.$inferInsert;

export type Account = typeof account.$inferSelect;
export type NewAccount = typeof account.$inferInsert;

export type Verification = typeof verification.$inferSelect;
export type NewVerification = typeof verification.$inferInsert;

export type Canvas = typeof canvas.$inferSelect;
export type NewCanvas = typeof canvas.$inferInsert;

export type AnalysisJob = typeof analysisJobs.$inferSelect;
export type NewAnalysisJob = typeof analysisJobs.$inferInsert;

// Analysis job status enum for type safety
export const AnalysisJobStatus = {
  PENDING: "pending",
  UPLOADING: "uploading", 
  ANALYZING: "analyzing",
  CRITIQUING: "critiquing",
  GENERATING: "generating",
  COMPLETED: "completed",
  FAILED: "failed",
} as const;

export type AnalysisJobStatusType = typeof AnalysisJobStatus[keyof typeof AnalysisJobStatus];

// MoodBoard for brand inspiration with array of images and single prompt
export const moodBoard = pgTable("moodboard", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id")
    .notNull()
    .references(() => user.id, { onDelete: "cascade" }),
  
  // Brand information
  brandName: text("brand_name").notNull(),
  brandSlogan: text("brand_slogan"),
  description: text("description"),
  colorPalette: jsonb("color_palette").$type<string[]>(), // Array of color codes
  
  // Images array and single prompt
  images: jsonb("images").$type<string[]>(), // Array of image URLs
  prompt: text("prompt"), // Single prompt for all images
  
  // Timestamps
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
});

// MoodBoardCanvas for storing generated canvas URLs (same order as moodboard images)
export const moodBoardCanvas = pgTable("moodboard_canvas", {
  id: uuid("id").primaryKey().defaultRandom(),
  moodboardId: uuid("moodboard_id")
    .notNull()
    .references(() => moodBoard.id, { onDelete: "cascade" }),
  
  // Canvas information
  name: text("name").notNull().default("Untitled Canvas"),
  canvasUrls: jsonb("canvas_urls").$type<string[]>().notNull(), // Array of canvas URLs in same order as moodboard images
  prompt: text("prompt").notNull(),
  
  // Additional metadata
  isFavorite: boolean("is_favorite")
    .$defaultFn(() => false)
    .notNull(),
  generationParams: jsonb("generation_params").$type<Record<string, any>>(),
  
  // Timestamps
  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
});

// Type exports for MoodBoard
export type MoodBoard = typeof moodBoard.$inferSelect;
export type NewMoodBoard = typeof moodBoard.$inferInsert;

export type MoodBoardCanvas = typeof moodBoardCanvas.$inferSelect;
export type NewMoodBoardCanvas = typeof moodBoardCanvas.$inferInsert;
