-- Add indexes to log_entries tables for run_id and machine_id
ALTER TABLE `default`.`log_entries` ADD INDEX `idx_machine_id_timestamp` ((machine_id, timestamp)) TYPE minmax GRANULARITY 1;
ALTER TABLE `default`.`log_entries` ADD INDEX `idx_run_id_timestamp` ((run_id, timestamp)) TYPE minmax GRANULARITY 1;
-- Add indexes to workflow_events tables for run_id and machine_id
ALTER TABLE `default`.`workflow_events` ADD INDEX `idx_machine_id_timestamp` ((machine_id, timestamp)) TYPE minmax GRANULARITY 1;
ALTER TABLE `default`.`workflow_events` ADD INDEX `idx_run_id_timestamp` ((run_id, timestamp)) TYPE minmax GRANULARITY 1;
