DO $$ BEGIN
 CREATE TYPE "gpu_provider" AS ENUM('modal');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "comfyui_deploy"."gpu_events" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"machine_id" uuid,
	"start_time" timestamp,
	"end_time" timestamp,
	"gpu" "machine_gpu",
	"gpu_provider" "gpu_provider" NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."gpu_events" ADD CONSTRAINT "gpu_events_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."gpu_events" ADD CONSTRAINT "gpu_events_machine_id_machines_id_fk" FOREIGN KEY ("machine_id") REFERENCES "comfyui_deploy"."machines"("id") ON DELETE set null ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
