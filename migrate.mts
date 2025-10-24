import { config } from "dotenv";
import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";

config({
  path: ".local.env",
});

const migrationsFolderName = process.env.MIGRATIONS_FOLDER || "drizzle";

const dbUrl = (process.env.DATABASE_URL || process.env.POSTGRES_URL || "").trim();

// Strip any query params (?ssl=true, ?sslmode=require, etc.)
let connectionString = dbUrl.split("?")[0];

let ssl: boolean | { rejectUnauthorized: false } = true;
try {
  const parsed = new URL(dbUrl);
  const host = parsed.hostname;

  // If user explicitly disables SSL via env, honor it
  if (
    process.env.SSL === "false" ||
    process.env.POSTGRES_SSL === "false" ||
    process.env.POSTGRES_SSL === "disable"
  ) {
    ssl = false;
  } else if (host.endsWith("railway.internal") || process.env.ALLOW_SELF_SIGNED === "true") {
    // Keep TLS, ignore Railway's self-signed cert on internal host
    ssl = { rejectUnauthorized: false };
  } else {
    // Default: enable TLS with verification when not using internal host
    ssl = true;
  }
} catch {
  // If URL parsing fails, fall back to safe default
  ssl = { rejectUnauthorized: false };
}

const migrationsFolderPath = new URL(`./${migrationsFolderName}`, import.meta.url).pathname;

const isDevContainer = process.env.REMOTE_CONTAINERS !== undefined;
if (isDevContainer)
  connectionString = connectionString.replace(
    "localhost",
    "host.docker.internal",
  );

const sql = postgres(connectionString, { max: 1, ssl });
const db = drizzle(sql, {
  // logger: true,
});

let retries = 20;
while (retries) {
  try {
    await sql`SELECT NOW()`;
    console.log("Database is live");
    break;
  } catch (error) {
    console.error("Database is not live yet at ", dbUrl, error);
    retries -= 1;
    console.log(`Retries left: ${retries}`);
    await new Promise((res) => setTimeout(res, 1000));
  }
}

console.log("Migrating... DB 1");
await migrate(db, { migrationsFolder: migrationsFolderPath });

console.log("Done!");
process.exit();
