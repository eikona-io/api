ALTER TABLE "comfyui_deploy"."workflows" ADD COLUMN "description" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."workflows" DROP COLUMN IF EXISTS "workflow_data";