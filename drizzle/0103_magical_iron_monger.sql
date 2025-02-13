ALTER TABLE "comfyui_deploy"."workflow_runs" ALTER COLUMN "workflow_id" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."workflow_runs" ADD COLUMN "workflow_api" jsonb;