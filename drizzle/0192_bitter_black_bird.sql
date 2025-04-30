ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "cpu_request" real;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "cpu_limit" real;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "memory_request" integer;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machine_versions" ADD COLUMN "memory_limit" integer;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "cpu_request" real;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "cpu_limit" real;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "memory_request" integer;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "memory_limit" integer;