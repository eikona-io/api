# Overview

- Clickhouse is gets started in `apps/web/docker-compose.yml`
- We use `apps/web/clickhouse/data` to hold your data, the contents of which are gitignored, except the folder itself.
- `atlas.hcl` is the configuration file for atlas,but we do not currently use it.
- `schema.sql` is the schema of the database.
- Our migration files live in `apps/web/clickhouse/migrations`.

# Setup
- Restart server, check clickhouse is up and running 
- Install clickhouse client
  - https://clickhouse.com/docs/en/install
- Run `./clickhouse client` https://clickhouse.com/docs/en/interfaces/cli
- Login into DB and verify clickhouse is running
- Password is `password`
- In console run: `show tables`

- You need to install and log into atlas.
    - Contact `nick` to get added to the team
    - We only use atlas to create the migration files
- Install atlas
    - `curl -sSf https://atlasgo.sh | sh ` 
    - https://atlasgo.io/getting-started/ (if not mac)
- Setup atlas
    - `Atlas login`
    - If this doesn’t work reach out to Nick you should be part of our organization
- Run migrations
    - `cd apps/web`
    - `bun run ch:migrate-local`
- If successful 
    - Run `show tables` in `./clickhouse client` console to verify

# Schema update local

- To update the schema, make the appropriate changes to `schema.sql`.
- You can then generate the migration using `bun run generate3`. This will modify the `apps/web/clickhouse/migrations` folder.
- After you can then apply it to your local database with `bun run ch:migrate-local`.

# Schema update Cloud

`atlas migrate apply --dir file://clickhouse/migrations -u "clickhouse://<user>:<password>@<subdomain>.us-east-2.aws.clickhouse.cloud:9440?secure=true"`

Run this in `web` dir

Above is an example command to apply the migration to a cloud database.
Currently this is done manually by us as the CI/CD is painful to setup and we do not expect to have many migrations in the short term.