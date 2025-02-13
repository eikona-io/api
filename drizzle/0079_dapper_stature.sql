ALTER TYPE "machine_type" ADD VALUE 'workspace';--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "pod_id" text;