ALTER TYPE "machine_status" ADD VALUE 'not-started';--> statement-breakpoint
COMMIT;
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "static_assets_status" "machine_status" DEFAULT 'not-started' NOT NULL;