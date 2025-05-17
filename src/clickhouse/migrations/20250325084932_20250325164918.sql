ALTER TABLE `default`.`log_entries` MODIFY COLUMN `log_level` Enum8('debug', 'info', 'warning', 'error', 'ws_event', 'builder', 'webhook');
