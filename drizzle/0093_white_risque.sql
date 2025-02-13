DO $$ BEGIN
 CREATE TYPE "machine_builder_version" AS ENUM('2', '3');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "machine_builder_version" "machine_builder_version" DEFAULT '2';