ALTER TABLE "comfyui_deploy"."api_keys" 
ADD COLUMN IF NOT EXISTS "scopes" JSONB,
ADD COLUMN IF NOT EXISTS "token_type" VARCHAR NOT NULL DEFAULT 'user' 
CHECK ("token_type" IN ('user', 'machine', 'scoped'));

DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'api_key_token_type') THEN
        CREATE TYPE "comfyui_deploy"."api_key_token_type" AS ENUM ('user', 'machine', 'scoped');
    END IF;
END $$;

UPDATE "comfyui_deploy"."api_keys" 
SET "token_type" = 'user' 
WHERE "token_type" IS NULL;
