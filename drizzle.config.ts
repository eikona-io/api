import { config } from "dotenv";
import type { Config } from "drizzle-kit";

config({
  path: `.env.local`,
});

const dbUrl = process.env.DATABASE_URL || process.env.POSTGRES_URL

export default {
  // schema: "./src/db/schema.ts",
  schema: "./schema.ts",
  // driver: "",
  dialect: "postgresql",
  out: "./drizzle",
  dbCredentials: {
    url:
      (dbUrl as string) +
      (process.env.POSTGRES_SSL !== "false" ? "?ssl=true" : ""),
  },
} satisfies Config;
