import { config } from "dotenv";
import type { Config } from "drizzle-kit";

config({ path: ".env.local" });

const dbUrl = process.env.DATABASE_URL || process.env.POSTGRES_URL;

export default {
  schema: "./schema.ts",
  dialect: "postgresql",
  out: "./drizzle",
  dbCredentials: {
    url: dbUrl as string,
    ssl: { rejectUnauthorized: false }, // 👈 works for Railway internal host
  },
} satisfies Config;