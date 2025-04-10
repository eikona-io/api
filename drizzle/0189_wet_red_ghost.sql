CREATE TABLE "comfyui_deploy"."machine_secrets" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"machine_id" uuid NOT NULL,
	"secret_id" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "unq_machine_secret" UNIQUE("machine_id","secret_id")
);
--> statement-breakpoint
CREATE TABLE "comfyui_deploy"."secrets" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"name" text NOT NULL,
	"environment_variables" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machine_secrets" ADD CONSTRAINT "machine_secrets_machine_id_machines_id_fk" FOREIGN KEY ("machine_id") REFERENCES "comfyui_deploy"."machines"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."machine_secrets" ADD CONSTRAINT "machine_secrets_secret_id_secrets_id_fk" FOREIGN KEY ("secret_id") REFERENCES "comfyui_deploy"."secrets"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."secrets" ADD CONSTRAINT "secrets_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "idx_machine_secrets_machine_id" ON "comfyui_deploy"."machine_secrets" USING btree ("machine_id");--> statement-breakpoint
CREATE INDEX "idx_machine_secrets_secret_id" ON "comfyui_deploy"."machine_secrets" USING btree ("secret_id");