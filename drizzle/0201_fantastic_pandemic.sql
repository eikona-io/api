ALTER TABLE "comfyui_deploy"."output_shares" ADD COLUMN "deployment_id" uuid;--> statement-breakpoint
ALTER TABLE "comfyui_deploy"."output_shares" ADD CONSTRAINT "output_shares_deployment_id_deployments_id_fk" FOREIGN KEY ("deployment_id") REFERENCES "comfyui_deploy"."deployments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "output_shares_deployment_id_index" ON "comfyui_deploy"."output_shares" USING btree ("deployment_id");--> statement-breakpoint
CREATE INDEX "output_shares_user_visibility_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("user_id","visibility","created_at");--> statement-breakpoint
CREATE INDEX "output_shares_visibility_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("visibility","created_at");--> statement-breakpoint
CREATE INDEX "output_shares_type_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("output_type","created_at");--> statement-breakpoint
CREATE INDEX "output_shares_deployment_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("deployment_id","created_at");