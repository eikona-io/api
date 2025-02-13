ALTER TABLE "comfyui_deploy"."user_settings" ADD COLUMN "account_settings" boolean DEFAULT false NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ADD COLUMN "workflow_limit" integer DEFAULT 2 NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."user_settings" ADD COLUMN "machine_limit" integer DEFAULT 0 NOT NULL;