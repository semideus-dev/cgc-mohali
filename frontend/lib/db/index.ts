import { drizzle } from "drizzle-orm/neon-http";
import { neon } from "@neondatabase/serverless";

import env from "@/lib/env";

if (!env.databaseUrl) {
  throw new Error("DATABASE_URL is not set");
}

const sql = neon(env.databaseUrl);
export const db = drizzle(sql);
