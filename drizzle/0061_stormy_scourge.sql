ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "target_workflow_id" uuid;--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."machines" ADD CONSTRAINT "machines_target_workflow_id_workflows_id_fk" FOREIGN KEY ("target_workflow_id") REFERENCES "comfyui_deploy"."workflows"("id") ON DELETE set null ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
