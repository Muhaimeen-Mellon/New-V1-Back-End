create extension if not exists "pgcrypto";

create table if not exists knowledge_logs (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  topic text,
  content text,
  source text,
  emotion_tag text,
  codex_impact text,
  codex_entry_id uuid,
  created_at timestamp with time zone default now(),
  is_deleted boolean default false,
  content_vector tsvector
);

alter table knowledge_logs
add constraint if not exists fk_codex_entry
foreign key (codex_entry_id)
references codex_entries(id)
on delete set null;

create index if not exists idx_knowledge_user on knowledge_logs(user_id);
create index if not exists idx_knowledge_topic on knowledge_logs(topic);
create index if not exists idx_knowledge_created_at on knowledge_logs(created_at desc);
create index if not exists idx_knowledge_content_search 
on knowledge_logs using GIN (content_vector);

create or replace function update_knowledge_vector()
returns trigger as $$
begin
  new.content_vector := to_tsvector('english', new.content);
  return new;
end;
$$ language plpgsql;

create trigger tsvectorupdate
before insert or update on knowledge_logs
for each row execute procedure update_knowledge_vector();
