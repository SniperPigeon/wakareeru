# Database Migrations

Place SQLite migration scripts here using a numeric prefix:

```text
001_add_image_review_columns.sql
002_add_crop_quality_metrics.sql
```

`pipeline.utils.init_db()` applies scripts in numeric order when their version is
greater than `PRAGMA user_version`. After each script succeeds, `user_version` is
advanced to that script number.

Keep migrations append-only. For a new database, `config/schema.sql` creates the
latest baseline schema; migrations are mainly for upgrading existing databases.

When the latest baseline schema already contains a newly migrated column, add this
directive to make the migration safe for a database whose version lags its schema:

```sql
-- migration_skip_if_column_exists: table_name.column_name
```
