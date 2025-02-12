DO $$ BEGIN
 CREATE TYPE "webhook_status" AS ENUM('success', 'failed', 'not-started', 'running');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
ALTER TABLE "comfyui_deploy"."workflow_runs" ADD COLUMN "webhook_status" "webhook_status";--> statement-breakpoint