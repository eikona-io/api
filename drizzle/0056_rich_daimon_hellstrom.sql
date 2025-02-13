ALTER TABLE "comfyui_deploy"."models" ADD COLUMN "file_hash_sha256" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."models" DROP COLUMN IF EXISTS "file_hash";