# Deploy the Permanent Public Demo on Render + Supabase

This deployment keeps the real FastAPI + `webui/` console online as a public demo and stores reviewer feedback in Supabase as **pending curation records**.

## 1. Create the Supabase feedback table

Open Supabase > SQL Editor and run:

```sql
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
```

Optional but useful index:

```sql
create index if not exists rag_feedback_review_idx
on rag_feedback (review_status, approved_for_rag, created_at);
```

## 2. Deploy on Render

1. Push this repository to GitHub.
2. Render > New > Web Service.
3. Connect the GitHub repository.
4. Use:

```bash
pip install -r requirements.txt
```

as build command.

5. Use:

```bash
uvicorn classification_api:app --host 0.0.0.0 --port $PORT
```

as start command.

6. Add environment variables in Render:

```text
PUBLIC_DEMO_MODE=true
USE_OLLAMA=false
USE_SUPABASE_FEEDBACK=true
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_FEEDBACK_TABLE=rag_feedback
```

Do not commit the real Supabase service key to GitHub.

## 3. Test the deployed demo

Open the Render URL, for example:

```text
https://your-app.onrender.com/
```

Run:

1. Paste Article
2. Classify article
3. Submit reviewer feedback
4. Check Supabase table `rag_feedback`

The new row should have:

```text
review_status = pending
approved_for_rag = false
```

That is intentional. Public feedback should not enter the RAG bank automatically.
