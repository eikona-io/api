ALTER TABLE "comfyui_deploy"."user_settings" ALTER COLUMN "workflow_limit" DROP DEFAULT;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ALTER COLUMN "workflow_limit" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ALTER COLUMN "machine_limit" DROP DEFAULT;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ALTER COLUMN "machine_limit" DROP NOT NULL;