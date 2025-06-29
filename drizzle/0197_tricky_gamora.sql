CREATE TYPE "public"."output_share_visibility" AS ENUM('private', 'public', 'link');--> statement-breakpoint
CREATE TYPE "public"."output_type" AS ENUM('image', 'video', '3d', 'other');--> statement-breakpoint
CREATE TABLE "comfyui_deploy"."output_shares" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"org_id" text,
	"run_id" uuid NOT NULL,
	"output_id" uuid NOT NULL,
	"output_data" jsonb,
	"output_type" "output_type" DEFAULT 'other' NOT NULL,
	"visibility" "output_share_visibility" DEFAULT 'private' NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "comfyui_deploy"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_run_id_workflow_runs_id_fk" FOREIGN KEY ("run_id") REFERENCES "comfyui_deploy"."workflow_runs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_output_id_workflow_run_outputs_id_fk" FOREIGN KEY ("output_id") REFERENCES "comfyui_deploy"."workflow_run_outputs"("id") ON DELETE no action ON UPDATE no action;