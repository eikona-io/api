CREATE TABLE IF NOT EXISTS "comfyui_deploy"."machine_versions" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"machine_id" uuid NOT NULL,
	"version" integer NOT NULL,
	"user_id" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"comfyui_version" text,
	"gpu" "machine_gpu",
	"docker_command_steps" jsonb,
	"allow_concurrent_inputs" integer DEFAULT 1,
	"concurrency_limit" integer DEFAULT 2,
	"install_custom_node_with_gpu" boolean DEFAULT false,
	"run_timeout" integer DEFAULT 300 NOT NULL,
	"idle_timeout" integer DEFAULT 60 NOT NULL,
	"extra_docker_commands" jsonb,
	"machine_builder_version" "machine_builder_version" DEFAULT '2',
	"base_docker_image" text,
	"python_version" text,
	"extra_args" text,
	"prestart_command" text,
	"keep_warm" integer DEFAULT 0 NOT NULL
);
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "machine_version_id" uuid;--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."machine_versions" ADD CONSTRAINT "machine_versions_machine_id_machines_id_fk" FOREIGN KEY ("machine_id") REFERENCES "comfyui_deploy"."machines"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."machine_versions" ADD CONSTRAINT "machine_versions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_machine_versions_machine_id" ON "comfyui_deploy"."machine_versions" USING btree (machine_id);--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."machines" ADD CONSTRAINT "machines_machine_version_id_machine_versions_id_fk" FOREIGN KEY ("machine_version_id") REFERENCES "comfyui_deploy"."machine_versions"("id") ON DELETE set null ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
