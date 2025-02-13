DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'comfyui_deploy' 
                   AND table_name = 'workflows' 
                   AND column_name = 'deleted') THEN
        ALTER TABLE "comfyui_deploy"."workflows" ADD COLUMN "deleted" boolean DEFAULT false NOT NULL;
    END IF;
END $$;