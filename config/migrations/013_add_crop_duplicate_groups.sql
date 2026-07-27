CREATE TABLE IF NOT EXISTS crop_duplicate_groups (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_key                  TEXT NOT NULL UNIQUE,
    source_sha1                TEXT NOT NULL,
    detector_model             TEXT NOT NULL,
    nms_iou_threshold          REAL NOT NULL,
    representative_crop_id     INTEGER NOT NULL REFERENCES crops(id),
    member_crop_ids_json       TEXT NOT NULL,
    candidate_labels_json      TEXT NOT NULL,
    member_count               INTEGER NOT NULL,
    candidate_label_count      INTEGER NOT NULL,
    box_x1                     REAL NOT NULL,
    box_y1                     REAL NOT NULL,
    box_x2                     REAL NOT NULL,
    box_y2                     REAL NOT NULL,
    global_top1_label          TEXT,
    global_top1_prob           REAL,
    global_top_k_json          TEXT NOT NULL DEFAULT '[]',
    proposed_label             TEXT,
    proposed_candidate_prob    REAL,
    proposed_candidate_margin  REAL,
    candidate_scores_json      TEXT NOT NULL DEFAULT '[]',
    proposal_model             TEXT,
    review_status              TEXT NOT NULL DEFAULT 'pending'
                               CHECK (review_status IN (
                                   'pending', 'auto_resolved', 'confirmed', 'excluded'
                               )),
    resolved_label             TEXT,
    review_note                TEXT,
    reviewed_at                TEXT,
    detected_at                TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                 TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_crop_duplicate_groups_status
    ON crop_duplicate_groups(review_status);
CREATE INDEX IF NOT EXISTS idx_crop_duplicate_groups_sha1
    ON crop_duplicate_groups(source_sha1);

PRAGMA user_version = 13;
