ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "machine_hash" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "machine_hash" text;