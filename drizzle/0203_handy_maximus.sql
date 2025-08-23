ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "models_to_cache" jsonb DEFAULT '[]'::jsonb;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "enable_gpu_memory_snapshot" boolean DEFAULT false;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "models_to_cache" jsonb DEFAULT '[]'::jsonb;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "enable_gpu_memory_snapshot" boolean DEFAULT false;