CREATE TABLE "comfyui_deploy"."shared_workflows" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"workflow_id" uuid NOT NULL,
	"workflow_version_id" uuid,
	"workflow_export" jsonb NOT NULL,
	"share_slug" text NOT NULL,
	"title" text NOT NULL,
	"description" text,
	"cover_image" text,
	"is_public" boolean DEFAULT true NOT NULL,
	"view_count" integer DEFAULT 0 NOT NULL,
	"download_count" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "shared_workflows_share_slug_unique" UNIQUE("share_slug")
);
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."shared_workflows" ADD CONSTRAINT "shared_workflows_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."shared_workflows" ADD CONSTRAINT "shared_workflows_workflow_id_workflows_id_fk" FOREIGN KEY ("workflow_id") REFERENCES "comfyui_deploy"."workflows"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."shared_workflows" ADD CONSTRAINT "shared_workflows_workflow_version_id_workflow_versions_id_fk" FOREIGN KEY ("workflow_version_id") REFERENCES "comfyui_deploy"."workflow_versions"("id") ON DELETE no action ON UPDATE no action;