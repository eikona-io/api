CREATE TABLE IF NOT EXISTS "comfyui_deploy"."credits" (
	"user_or_org_id" text PRIMARY KEY NOT NULL,
	"ws_credit" real DEFAULT 100 NOT NULL,
	"last_updated" timestamp DEFAULT now() NOT NULL
);
