-- Table for status updates
CREATE TABLE status_updates (
    event_id UUID,
    run_id UUID,
    timestamp DateTime64(3),
    status Enum8('not-started' = 1, 'queued' = 2, 'started' = 3, 'running' = 4, 'uploading' = 5, 'success' = 6, 'failed' = 7, 'timeout' = 8, 'cancelled' = 9)
) ENGINE = MergeTree()
ORDER BY (run_id, timestamp)
PARTITION BY toYYYYMM(timestamp);

-- Table for progress updates
CREATE TABLE progress_updates (
    event_id UUID,
    run_id UUID,
    timestamp DateTime64(3),
    progress Float32
) ENGINE = MergeTree()
ORDER BY (run_id, timestamp)
PARTITION BY toYYYYMM(timestamp);

-- Table for log entries
CREATE TABLE log_entries (
    event_id UUID,
    run_id UUID,
    timestamp DateTime64(3),
    log_level Enum8('debug' = 1, 'info' = 2, 'warning' = 3, 'error' = 4),
    message String
) ENGINE = MergeTree()
ORDER BY (run_id, timestamp)
PARTITION BY toYYYYMM(timestamp);

-- Materialized view for status updates
CREATE MATERIALIZED VIEW status_updates_mv
ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (run_id, timestamp)
POPULATE
AS SELECT
    run_id,
    'status_update' AS event_type,
    timestamp,
    status AS event_data
FROM status_updates;

-- Materialized view for progress updates
CREATE MATERIALIZED VIEW progress_updates_mv
ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (run_id, timestamp)
POPULATE
AS SELECT
    run_id,
    'progress_update' AS event_type,
    timestamp,
    toString(progress) AS event_data
FROM progress_updates;

-- Materialized view for log entries
CREATE MATERIALIZED VIEW log_entries_mv
ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (run_id, timestamp)
POPULATE
AS SELECT
    run_id,
    'log_entry' AS event_type,
    timestamp,
    message AS event_data
FROM log_entries;

-- Create a view to combine all updates
CREATE VIEW latest_updates_view AS
SELECT run_id, event_type, timestamp, event_data
FROM (
    SELECT * FROM status_updates_mv
    UNION ALL
    SELECT * FROM progress_updates_mv
    UNION ALL
    SELECT * FROM log_entries_mv
)
ORDER BY timestamp DESC;


-- Step 1: Drop the existing views
DROP VIEW IF EXISTS latest_updates_view;

-- Drop the materialized view if it exists (using a workaround)
CREATE TABLE IF NOT EXISTS status_updates_mv_tmp (dummy UInt8) ENGINE = Memory;
DROP TABLE status_updates_mv_tmp;
DROP TABLE IF EXISTS status_updates_mv;

-- Step 2: Alter the status_updates table
ALTER TABLE status_updates
    MODIFY COLUMN status Enum8('not-started' = 1, 'queued' = 2, 'started' = 3, 'running' = 4, 'uploading' = 5, 'success' = 6, 'failed' = 7, 'timeout' = 8, 'cancelled' = 9);

-- Step 3: Recreate the materialized view for status updates
CREATE MATERIALIZED VIEW status_updates_mv
ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (run_id, timestamp)
POPULATE
AS SELECT
    run_id,
    'status_update' AS event_type,
    timestamp,
    status AS event_data
FROM status_updates;

-- Step 4: Recreate the combined view
CREATE VIEW latest_updates_view AS
SELECT run_id, event_type, timestamp, event_data
FROM (
    SELECT * FROM status_updates_mv
    UNION ALL
    SELECT * FROM progress_updates_mv
    UNION ALL
    SELECT * FROM log_entries_mv
)
ORDER BY timestamp DESC;