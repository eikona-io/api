CREATE TABLE IF NOT EXISTS "comfyui_deploy"."assets" (
	"id" text PRIMARY KEY NOT NULL,
	"user_id" text,
	"org_id" text,
	"name" text NOT NULL,
	"is_folder" boolean DEFAULT false NOT NULL,
	"path" text DEFAULT '/' NOT NULL,
	"file_size" bigint,
	"url" text,
	"mime_type" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"deleted" boolean DEFAULT false NOT NULL
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."assets" ADD CONSTRAINT "assets_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_path" ON "comfyui_deploy"."assets" USING btree (path);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_user_path" ON "comfyui_deploy"."assets" USING btree (user_id,path);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_org_path" ON "comfyui_deploy"."assets" USING btree (org_id,path);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_user_org_path" ON "comfyui_deploy"."assets" USING btree (user_id,org_id,path);