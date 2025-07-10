CREATE INDEX "output_shares_user_visibility_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("user_id","visibility","created_at");--> statement-breakpoint
CREATE INDEX "output_shares_visibility_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("visibility","created_at");--> statement-breakpoint
CREATE INDEX "output_shares_type_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("output_type","created_at");--> statement-breakpoint
CREATE INDEX "output_shares_deployment_created_idx" ON "comfyui_deploy"."output_shares" USING btree ("deployment_id","created_at");