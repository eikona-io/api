ALTER TABLE `default`.`log_entries` MODIFY TTL toDateTime(timestamp) + toIntervalDay(30);
ALTER TABLE `default`.`workflow_events` MODIFY TTL toDateTime(timestamp) + toIntervalDay(30);
