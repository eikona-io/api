ALTER TABLE "comfyui_deploy"."output_shares" DROP COLUMN "shared_output_ids";--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD COLUMN "output_id" uuid NOT NULL;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD COLUMN "output_data" jsonb;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_output_id_workflow_run_outputs_id_fk" FOREIGN KEY ("output_id") REFERENCES "comfyui_deploy"."workflow_run_outputs"("id") ON DELETE no action ON UPDATE no action;
