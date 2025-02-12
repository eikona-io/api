ALTER TABLE "comfyui_deploy"."deployments" ADD COLUMN "machine_version_id" uuid;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."deployments" ADD COLUMN "modal_image_id" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."deployments" ADD COLUMN "gpu" "machine_gpu";--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."deployments" ADD COLUMN "run_timeout" integer DEFAULT 300 NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."deployments" ADD COLUMN "idle_timeout" integer DEFAULT 0 NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."deployments" ADD COLUMN "keep_warm" integer DEFAULT 0 NOT NULL;