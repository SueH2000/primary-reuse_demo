create table if not exists rag_feedback (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),

  paper_id text,
  identifier text,
  title text,

  predicted_label text,
  corrected_label text,
  is_correct boolean,

  evidence_sentence text,
  reviewer text,
  reviewer_email text,
  reviewer_note text,

  input_text text,
  result_json jsonb,
  local_feedback_row jsonb,

  review_status text default 'pending',
  curator_note text,
  reviewed_at timestamptz,
  approved_for_rag boolean default false,

  submitted_at_utc timestamptz,
  source_app text,
  consent_to_store_input_text boolean default false
);

create index if not exists rag_feedback_review_idx
on rag_feedback (review_status, approved_for_rag, created_at);
