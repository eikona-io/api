CREATE INDEX IF NOT EXISTS "deployments_updated_at_index" ON "comfyui_deploy"."deployments" USING btree (updated_at);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "deployments_modal_image_id_index" ON "comfyui_deploy"."deployments" USING btree (modal_image_id);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "deployments_environment_index" ON "comfyui_deploy"."deployments" USING btree (environment);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "deployments_user_id_index" ON "comfyui_deploy"."deployments" USING btree (user_id);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "deployments_org_id_index" ON "comfyui_deploy"."deployments" USING btree (org_id);