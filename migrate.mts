import { config } from "dotenv";
import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";

config({
  path: ".local.env",
});

const migrationsFolderName = process.env.MIGRATIONS_FOLDER || "drizzle";
let sslMode: string | boolean = process.env.SSL || "require";

if (sslMode === "false") sslMode = false;

const dbUrl = process.env.DATABASE_URL || process.env.POSTGRES_URL

let connectionString = dbUrl!;

console.log(connectionString)

const isDevContainer = process.env.REMOTE_CONTAINERS !== undefined;
if (isDevContainer)
  connectionString = connectionString.replace(
    "localhost",
    "host.docker.internal",
  );

const sql = postgres(connectionString, { max: 1, ssl: sslMode as any });
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
await migrate(db, { migrationsFolder: migrationsFolderName });

console.log("Done!");
process.exit();
