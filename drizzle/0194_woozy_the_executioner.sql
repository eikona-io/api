CREATE TYPE "public"."api_key_token_type" AS ENUM('user', 'machine', 'scoped');--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."api_keys" ADD COLUMN "scopes" jsonb;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."api_keys" ADD COLUMN "token_type" "api_key_token_type" DEFAULT 'user' NOT NULL;