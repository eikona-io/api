DO $$ BEGIN
 CREATE TYPE "public"."output_visibility" AS ENUM('public', 'private');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ADD COLUMN "output_visibility" "output_visibility" DEFAULT 'public';