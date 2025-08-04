ALTER TABLE `default`.`log_entries` DROP INDEX `idx_machine_id_timestamp`;
ALTER TABLE `default`.`log_entries` DROP INDEX `idx_run_id_timestamp`;
ALTER TABLE `default`.`workflow_events` DROP INDEX `idx_machine_id_timestamp`;
ALTER TABLE `default`.`workflow_events` DROP INDEX `idx_run_id_timestamp`;
