import { config } from "dotenv";

config({
  path: ".local.env",
});

const CLICKHOUSE_HOST = process.env.CLICKHOUSE_HOST || "localhost";
const CLICKHOUSE_PORT = process.env.CLICKHOUSE_PORT || "8123";
const CLICKHOUSE_USER = process.env.CLICKHOUSE_USER || "default";
const CLICKHOUSE_PASSWORD = process.env.CLICKHOUSE_PASSWORD || "password";
const CLICKHOUSE_DATABASE = process.env.CLICKHOUSE_DATABASE || "default";

const CLICKHOUSE_URL = `http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}`;

async function testClickHouseConnection(): Promise<void> {
  console.log(`Testing ClickHouse connection to: ${CLICKHOUSE_URL}`);
  console.log(`Database: ${CLICKHOUSE_DATABASE}, User: ${CLICKHOUSE_USER}`);

  try {
    const url = new URL(CLICKHOUSE_URL);
    url.searchParams.set("user", CLICKHOUSE_USER);
    url.searchParams.set("password", CLICKHOUSE_PASSWORD);
    url.searchParams.set("database", CLICKHOUSE_DATABASE);

    const response = await fetch(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: "SELECT version()",
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ClickHouse connection failed: ${response.status} ${response.statusText}\n${errorText}`);
    }

    const version = await response.text();
    console.log(`✅ ClickHouse connection successful! Version: ${version.trim()}`);

    // Test showing tables
    const tablesResponse = await fetch(url.toString(), {
      method: "POST",
      body: "SHOW TABLES",
    });

    if (tablesResponse.ok) {
      const tables = await tablesResponse.text();
      console.log(`📋 Existing tables:\n${tables || "(no tables)"}`);
    }

  } catch (error) {
    console.error("❌ ClickHouse connection failed:", error);
    process.exit(1);
  }
}

testClickHouseConnection(); 