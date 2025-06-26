CREATE TYPE "public"."output_share_visibility" AS ENUM('link-only', 'public', 'public-in-org');--> statement-breakpoint
CREATE TABLE "comfyui_deploy"."output_shares" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"run_id" uuid NOT NULL,
	"shared_output_ids" jsonb,
	"share_slug" text NOT NULL,
	"visibility" "output_share_visibility" DEFAULT 'link-only' NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "unique_output_share_slug" UNIQUE("share_slug")
);
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_run_id_workflow_runs_id_fk" FOREIGN KEY ("run_id") REFERENCES "comfyui_deploy"."workflow_runs"("id") ON DELETE no action ON UPDATE no action;