ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "python_version" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "extra_args" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "prestart_command" text;