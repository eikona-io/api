ALTER TABLE "comfyui_deploy"."user_settings" ALTER COLUMN "workflow_limit" SET DATA TYPE real;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ALTER COLUMN "machine_limit" SET DATA TYPE real;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" DROP COLUMN IF EXISTS "account_settings";