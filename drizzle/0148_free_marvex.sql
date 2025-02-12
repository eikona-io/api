ALTER TABLE "comfyui_deploy"."gpu_events" ADD COLUMN "cost_item_title" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."gpu_events" ADD COLUMN "cost" real DEFAULT 0;