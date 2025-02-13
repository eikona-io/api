DO $$ BEGIN
 CREATE TYPE "workspace_machine_gpu" AS ENUM('4090');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "ws_gpu" "workspace_machine_gpu";--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "base_docker_image" text;