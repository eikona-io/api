ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "retrieve_static_assets" boolean DEFAULT false;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "filename_list_cache" jsonb;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "extensions" jsonb;--> statement-breakpoint