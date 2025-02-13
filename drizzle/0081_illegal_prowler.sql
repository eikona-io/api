ALTER TYPE "gpu_provider" ADD VALUE 'runpod';--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."gpu_events" ADD COLUMN "ws_gpu" "workspace_machine_gpu";