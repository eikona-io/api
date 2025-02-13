ALTER TABLE "comfyui_deploy"."workflows" ADD COLUMN "selected_machine_id" uuid;--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "comfyui_deploy"."workflows" ADD CONSTRAINT "workflows_selected_machine_id_machines_id_fk" FOREIGN KEY ("selected_machine_id") REFERENCES "comfyui_deploy"."machines"("id") ON DELETE set null ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
