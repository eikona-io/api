ALTER TABLE "comfyui_deploy"."workflow_versions" ADD COLUMN "user_id" text;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."workflow_versions" ADD COLUMN "comment" text;--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."workflow_versions" ADD CONSTRAINT "workflow_versions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE set null ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
