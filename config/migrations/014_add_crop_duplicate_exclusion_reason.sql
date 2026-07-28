PRAGMA foreign_keys = OFF;

ALTER TABLE crop_duplicate_groups RENAME TO crop_duplicate_groups_v13;

CREATE TABLE crop_duplicate_groups (
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
    exclusion_reason           TEXT
                               CHECK (
                                   exclusion_reason IS NULL
                                   OR exclusion_reason IN (
                                       'bad_crop', 'out_of_label_space'
                                   )
                               ),
    review_note                TEXT,
    reviewed_at                TEXT,
    detected_at                TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                 TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO crop_duplicate_groups (
    id,
    group_key,
    source_sha1,
    detector_model,
    nms_iou_threshold,
    representative_crop_id,
    member_crop_ids_json,
    candidate_labels_json,
    member_count,
    candidate_label_count,
    box_x1,
    box_y1,
    box_x2,
    box_y2,
    global_top1_label,
    global_top1_prob,
    global_top_k_json,
    proposed_label,
    proposed_candidate_prob,
    proposed_candidate_margin,
    candidate_scores_json,
    proposal_model,
    review_status,
    resolved_label,
    review_note,
    reviewed_at,
    detected_at,
    updated_at
)
SELECT
    id,
    group_key,
    source_sha1,
    detector_model,
    nms_iou_threshold,
    representative_crop_id,
    member_crop_ids_json,
    candidate_labels_json,
    member_count,
    candidate_label_count,
    box_x1,
    box_y1,
    box_x2,
    box_y2,
    global_top1_label,
    global_top1_prob,
    global_top_k_json,
    proposed_label,
    proposed_candidate_prob,
    proposed_candidate_margin,
    candidate_scores_json,
    proposal_model,
    review_status,
    resolved_label,
    review_note,
    reviewed_at,
    detected_at,
    updated_at
FROM crop_duplicate_groups_v13;

DROP TABLE crop_duplicate_groups_v13;

CREATE INDEX idx_crop_duplicate_groups_status
    ON crop_duplicate_groups(review_status);
CREATE INDEX idx_crop_duplicate_groups_sha1
    ON crop_duplicate_groups(source_sha1);

PRAGMA foreign_key_check;
PRAGMA foreign_keys = ON;
PRAGMA user_version = 14;
