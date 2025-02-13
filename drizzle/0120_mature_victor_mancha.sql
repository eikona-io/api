DO $$ BEGIN
 CREATE TYPE "public"."training_type" AS ENUM('flux-lora');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "comfyui_deploy"."trainings" (
	"id" text PRIMARY KEY NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"name" text NOT NULL,
	"inputs" jsonb,
	"outputs" jsonb,
	"status" "resource_upload" DEFAULT 'started' NOT NULL,
	"type" "training_type" NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."trainings" ADD CONSTRAINT "trainings_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
