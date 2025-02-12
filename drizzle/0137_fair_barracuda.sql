DO $$ BEGIN
 CREATE TYPE "public"."api_version" AS ENUM('v1', 'v2');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ADD COLUMN "api_version" "api_version" DEFAULT 'v1';