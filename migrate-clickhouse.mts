import { config } from "dotenv";
import { readdir, readFile } from "fs/promises";
import { join } from "path";

config({
  path: ".local.env",
});

// ClickHouse connection settings from environment or defaults for Docker
const CLICKHOUSE_HOST = process.env.CLICKHOUSE_HOST || "localhost";
const CLICKHOUSE_PORT = process.env.CLICKHOUSE_PORT || "8123";
const CLICKHOUSE_USER = process.env.CLICKHOUSE_USER || "default";
const CLICKHOUSE_PASSWORD = process.env.CLICKHOUSE_PASSWORD || "password";
const CLICKHOUSE_DATABASE = process.env.CLICKHOUSE_DATABASE || "default";

const CLICKHOUSE_URL = `http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}`;
const MIGRATIONS_DIR = "./src/clickhouse/migrations";

interface MigrationFile {
  filename: string;
  timestamp: string;
  content: string;
}

async function executeClickHouseQuery(query: string, allowErrors: string[] = []): Promise<any> {
  const url = new URL(CLICKHOUSE_URL);
  url.searchParams.set("user", CLICKHOUSE_USER);
  url.searchParams.set("password", CLICKHOUSE_PASSWORD);
  url.searchParams.set("database", CLICKHOUSE_DATABASE);

  console.log(`Executing query: ${query.substring(0, 100)}${query.length > 100 ? "..." : ""}`);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: query,
  });

  if (!response.ok) {
    const errorText = await response.text();
    
    // Check if this is an allowed error (like column already exists/doesn't exist)
    for (const allowedError of allowErrors) {
      if (errorText.includes(allowedError)) {
        console.log(`⚠️  Ignoring expected error: ${allowedError}`);
        return "";
      }
    }
    
    throw new Error(`ClickHouse query failed: ${response.status} ${response.statusText}\n${errorText}`);
  }

  return response.text();
}

async function columnExists(table: string, column: string): Promise<boolean> {
  try {
    const result = await executeClickHouseQuery(`
      SELECT name FROM system.columns 
      WHERE table = '${table}' AND database = '${CLICKHOUSE_DATABASE}' AND name = '${column}'
    `);
    return result.trim() !== "";
  } catch (error) {
    console.log(`Warning: Could not check if column exists: ${error}`);
    return false;
  }
}

async function tableExists(table: string): Promise<boolean> {
  try {
    const result = await executeClickHouseQuery(`
      SELECT name FROM system.tables 
      WHERE database = '${CLICKHOUSE_DATABASE}' AND name = '${table}'
    `);
    return result.trim() !== "";
  } catch (error) {
    console.log(`Warning: Could not check if table exists: ${error}`);
    return false;
  }
}

async function createMigrationsTable(): Promise<void> {
  const createTableQuery = `
    CREATE TABLE IF NOT EXISTS migrations (
      version String,
      applied_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY version
  `;

  await executeClickHouseQuery(createTableQuery);
  console.log("✅ Migrations table created or already exists");
}

async function getAppliedMigrations(): Promise<Set<string>> {
  try {
    const result = await executeClickHouseQuery("SELECT version FROM migrations ORDER BY version");
    const versions = result
      .trim()
      .split("\n")
      .filter((line: string) => line.trim() !== "")
      .map((line: string) => line.trim());
    
    return new Set(versions);
  } catch (error) {
    console.log("No migrations table found, will create it");
    return new Set();
  }
}

async function markMigrationAsApplied(version: string): Promise<void> {
  const insertQuery = `INSERT INTO migrations (version) VALUES ('${version}')`;
  await executeClickHouseQuery(insertQuery);
}

async function getMigrationFiles(): Promise<MigrationFile[]> {
  try {
    const files = await readdir(MIGRATIONS_DIR);
    const migrationFiles: MigrationFile[] = [];

    for (const filename of files) {
      if (filename.endsWith(".sql")) {
        const timestamp = filename.split("_")[0];
        const content = await readFile(join(MIGRATIONS_DIR, filename), "utf-8");
        migrationFiles.push({ filename, timestamp, content });
      }
    }

    // Sort by timestamp
    migrationFiles.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    return migrationFiles;
  } catch (error) {
    console.error("Error reading migration files:", error);
    throw error;
  }
}

async function waitForClickHouse(): Promise<void> {
  console.log("Waiting for ClickHouse to be ready...");
  let retries = 30;
  
  while (retries > 0) {
    try {
      await executeClickHouseQuery("SELECT 1");
      console.log("✅ ClickHouse is ready");
      return;
    } catch (error) {
      console.log(`ClickHouse not ready yet, retries left: ${retries}`);
      retries--;
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
  
  throw new Error("ClickHouse failed to become ready");
}

async function runMigrations(): Promise<void> {
  try {
    // Wait for ClickHouse to be ready
    await waitForClickHouse();

    // Create migrations table if it doesn't exist
    await createMigrationsTable();

    const tables = await executeClickHouseQuery("SHOW TABLES");
    console.log(tables);

    // Get already applied migrations
    const appliedMigrations = await getAppliedMigrations();
    console.log(`Found ${appliedMigrations.size} already applied migrations`);

    // Get all migration files
    const migrationFiles = await getMigrationFiles();
    console.log(`Found ${migrationFiles.length} migration files`);

    // console.log(migrationFiles);

    let appliedCount = 0;

    for (const migration of migrationFiles) {
      const migrationId = migration.filename.replace(".sql", "");
      
      if (appliedMigrations.has(migrationId)) {
        console.log(`⏭️  Skipping already applied migration: ${migration.filename}`);
        continue;
      }

      console.log(`🔄 Applying migration: ${migration.filename}`);

      // Split content by lines first, then reconstruct statements
      const lines = migration.content
        .split("\n")
        .map(line => line.trim())
        .filter(line => line.length > 0 && !line.startsWith("--"));
      
      // Reconstruct statements by joining lines until we hit a semicolon
      const statements: string[] = [];
      let currentStatement = "";
      
      for (const line of lines) {
        currentStatement += line + " ";
        if (line.endsWith(";")) {
          statements.push(currentStatement.trim().slice(0, -1)); // Remove trailing semicolon
          currentStatement = "";
        }
      }
      
      // Add any remaining statement (in case last statement doesn't end with semicolon)
      if (currentStatement.trim()) {
        statements.push(currentStatement.trim());
      }

      for (const statement of statements) {
        // console.log(statement);
        if (statement.trim()) {
          await executeClickHouseQuery(statement);
        }
      }

      await markMigrationAsApplied(migrationId);
      appliedCount++;
      console.log(`✅ Successfully applied migration: ${migration.filename}`);
    }

    console.log(`\n🎉 Migration complete! Applied ${appliedCount} new migrations.`);

  } catch (error) {
    console.error("❌ Migration failed:", error);
    process.exit(1);
  }
}

// Run migrations
runMigrations(); 