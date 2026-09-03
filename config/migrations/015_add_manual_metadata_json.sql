-- migration_skip_if_column_exists: images.manual_metadata_json
ALTER TABLE images
ADD COLUMN manual_metadata_json TEXT NOT NULL DEFAULT '{}';

PRAGMA user_version = 15;
