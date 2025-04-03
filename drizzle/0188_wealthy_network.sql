CREATE TABLE IF NOT EXISTS "comfyui_deploy"."machine_secrets" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"machine_id" uuid NOT NULL,
	"secret_id" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "unq_machine_secret" UNIQUE("machine_id","secret_id")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "comfyui_deploy"."secrets" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"environment_variables" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT FROM information_schema.columns 
    WHERE table_schema = 'comfyui_deploy' AND table_name = 'machines' AND column_name = 'optimized_runner'
  ) THEN
    ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "optimized_runner" boolean DEFAULT false NOT NULL;
  END IF;
END
$$;
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."machine_secrets" ADD CONSTRAINT "machine_secrets_machine_id_machines_id_fk" FOREIGN KEY ("machine_id") REFERENCES "comfyui_deploy"."machines"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."machine_secrets" ADD CONSTRAINT "machine_secrets_secret_id_secrets_id_fk" FOREIGN KEY ("secret_id") REFERENCES "comfyui_deploy"."secrets"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."secrets" ADD CONSTRAINT "secrets_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_machine_secrets_machine_id" ON "comfyui_deploy"."machine_secrets" USING btree (machine_id);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_machine_secrets_secret_id" ON "comfyui_deploy"."machine_secrets" USING btree (secret_id);