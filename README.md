# Motor Insurance Agentic AI -- Customer Support Resolution Agent

IITM Agentic AI capstone (Scenario 3: Customer Support). See the Problem
Framing Document for persona, workflow, and success criteria.

## Current build status

- [x] Directory structure
- [x] `core/settings.py` -- typed config from `.env`
- [x] `core/logging_config.py` + `core/observability.py` -- file logging + PII-safe transaction logging
- [x] `models/` -- 9-table SQLAlchemy schema (Phase 1-2 groundwork)
- [x] `repositories/` -- CRUD + domain queries per table (memory lookup, policy validity check, escalation lifecycle)
- [x] `knowledge_base/` -- 2 synthetic motor policy documents (comprehensive + third-party) ready for RAG ingestion
- [x] `services/seed_data.py` + `scripts/seed_customers.py` -- 20 synthetic customers with policies, 5 human support agents
- [x] `integrations/rag.py` + `integrations/policy_parser.py` -- knowledge_base ingestion into Pinecone
- [x] `api/` -- FastAPI app with `/health`, `POST /ingest`, and full `/tickets` CRUD + `POST /tickets/{id}/process`
- [x] `graph/` -- LangGraph agent core (classify -> tool-calling reasoning -> summarize -> faithfulness check -> rule-based escalation gate)
- [x] LangSmith tracing wired in (`core/tracing.py`)
- [x] `app.py` + `ui/` -- Streamlit UI (Customer Portal + Agent Console, hardcoded demo login)
- [ ] `integrations/memory.py` -- semantic memory (currently just repository queries; Pinecone customer-memory index not yet used)
- [x] `docker-compose.yml` -- Postgres for local dev

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                 # installs everything from pyproject.toml / uv.lock
cp .env.example .env    # fill in real OpenAI / Pinecone keys and DB url
```

## Running the tests

Tests run against an in-memory SQLite DB by default -- no Docker/Postgres
needed. `test_models.py` checks the schema/relationships; `test_repositories.py`
exercises the actual query methods the agent's tools will call (memory
lookup across a repeat customer's tickets, active-policy validity check,
escalation lifecycle).

```bash
uv run pytest tests/ -v
```

To run the exact same suite against your real Postgres (docker-compose) instance:

```bash
TEST_DATABASE_URL="postgresql+psycopg2://agent:localdev@localhost:5432/insurance_support" \
    uv run pytest tests/ -v
```

## Repository layer

One repository class per table in `customer_support_agent/repositories/`,
each wrapping a `Session` passed in by the caller (a FastAPI dependency
later, the test fixture for now). Beyond generic CRUD (`BaseRepository`),
the methods worth knowing about:

| Repository | Key method | Why it exists |
|---|---|---|
| `CustomerPolicyRepository` | `get_active_policy_for_customer()` | The validity check the agent should run *before* reasoning about coverage clauses |
| `InteractionRepository` | `get_recent_for_customer()` | The actual "memory" query -- joins through `Ticket`, not a separate memory table |
| `EscalationRepository` | `create()` / `update_status()` | Backs the `escalate_to_agent` tool and its lifecycle |
| `AgentRepository` | `get_or_create_by_name()` | Idempotent lookup so seeding/demo scripts don't duplicate agents |


## Switching to real Postgres (Docker Desktop)

Once `docker-compose.yml` is added (next step), bring up Postgres with:

```bash
docker compose up -d postgres
```

Then in `.env`, set:

```
DATABASE_URL=postgresql+psycopg2://agent:localdev@localhost:5432/insurance_support
```

Everything in `models/` and `repositories/` is written against the
SQLAlchemy ORM, so no code changes are needed -- only the connection string.

## Schema overview

| Table | Purpose |
|---|---|
| `customers` | Real customer PII (name, contact) -- system of record |
| `policy_documents` | Metadata for policy PDFs chunked into the Pinecone `policies` index |
| `customer_policies` | A specific customer's coverage instance (dates, status, premium) |
| `tickets` | One row per support ticket |
| `ticket_messages` | Multi-turn thread within a ticket |
| `interactions` | AI reasoning/audit trail: summary, cited clauses (JSON), faithfulness check, escalation flag |
| `escalations` | Human-side workflow lifecycle for an escalated case |
| `feedback` | Human agent feedback on an AI interaction (edit distance, rating) |
| `agents` | Human support agents referenced by escalations/feedback |

Design notes worth remembering:

- **PII lives in the DB, not in logs/traces.** `customers.name`/`contact_no`
  are real data -- the agent needs them to draft a reply. The
  "must not store personal data in logs" safety requirement is enforced at
  the LangSmith tracing boundary (redact before trace), not by hiding data
  in the schema.
- **`interactions` is not a "memory table."** It's the AI's audit trail.
  Memory (recalling a repeat customer's history) is a *query pattern* over
  `tickets` + `interactions`, not a separate table.
- **`policy_documents` vs `customer_policies`**: the former is the PDF
  template embedded into Pinecone; the latter is one customer's actual,
  dated coverage instance. `CustomerPolicy.is_valid_on(date)` is a cheap
  validity check the agent should run before reasoning about coverage
  clauses at all.

## Application logging

`core/logging_config.py` sets up a rotating file handler (`logs/app.log`,
5MB per file, 5 backups) plus console output. `core/observability.py`
provides `log_transaction(action, **context)`, a context manager that wraps
any unit of work (processing a ticket, seeding data, an API request later)
and writes one `START` + one `SUCCESS`/`FAILURE` line per transaction.

**PII redaction happens automatically**: any context key in `PII_FIELDS`
(`name`, `contact_no`, `contact_email`, `phone`, `email`, `vehicle_reg_no`)
is replaced with `[REDACTED]` before it's written, recursively through
nested dicts/lists. This is the actual enforcement point for the
"must not store personal data in logs" safety requirement -- see
`tests/test_logging.py` for the proof (a customer name goes in, never comes
out in the log file).

Usage in your own code:
```python
from customer_support_agent.core import log_transaction

with log_transaction("process_ticket", ticket_id=42, customer_id=7):
    ...  # do the work; name/contact fields in kwargs would be redacted automatically
```

## Knowledge base (RAG source documents)

`knowledge_base/` contains two **synthetic** (not real insurer content)
motor policy documents in Markdown, written with clause IDs that match the
seed/test data:

- `motor_comprehensive_policy_v3_2.md` -- own damage, theft, glass, rental
  cover, NCB, claims process, exclusions
- `motor_third_party_policy_v1_0.md` -- third-party liability only, no own
  damage/glass/rental cover

These are what Phase 4 (RAG ingestion) will chunk and embed into the
Pinecone `policies` index.

### Chunking strategy: clause-boundary first, LangChain splitter as a safety net

The primary chunk boundary is each document's own `### Clause X.X:` /
`#### Clause X.X:` heading (see `parse_clauses()` in
`integrations/policy_parser.py`) -- not a fixed character count. For a
structured policy document, one clause is already the right retrieval and
citation unit; a generic size-based splitter would either cut a clause
mid-sentence or merge unrelated clauses into one chunk, both of which hurt
citation accuracy (the "must not fabricate policies" safety requirement
depends on clean, traceable citations).

That said, nothing guarantees every clause stays short forever, so
LangChain's `RecursiveCharacterTextSplitter` is used as a **safety net**,
not the primary strategy: `split_oversized_chunk()` leaves any clause under
1200 characters untouched, and only sub-splits clauses that exceed it
(suffixing the clause_id, e.g. `OD-4.2` -> `OD-4.2-p1`, `OD-4.2-p2`, ...) so
long clauses don't get embedded as one diluted chunk or dropped/truncated.
Embeddings themselves go through `langchain-openai`'s `OpenAIEmbeddings`
rather than a raw OpenAI client call, so LangChain is genuinely part of the
ingestion path, consistent with the project's Track A framework choice.

## Seed data (20 synthetic customers)

```bash
uv run python scripts/seed_customers.py           # creates 20 customers + policies
uv run python scripts/seed_customers.py --force    # wipes existing customers first
uv run python scripts/seed_customers.py --count 10 --seed 7
```

Deterministic given the same `--seed` (default 42). Status distribution
across the 20 customers is deliberately not all-active: 14 active, 3
expired, 2 lapsed, 1 cancelled -- so the policy-validity check has real
negative cases to demonstrate, matching the "lapsed policy" failure case
called out in the Problem Framing Document.

`customer_support_agent/services/seed_data.py` holds the reusable logic;
`scripts/seed_customers.py` is the thin CLI wrapper (reads `DATABASE_URL`
from `.env`, guards against accidental double-seeding, prints a summary
table).

Running the script also seeds **5 human support agents** (idempotent --
safe to call every run), so `Escalation.assigned_agent_id` has real records
to point at:

| Name | Role |
|---|---|
| Priya Menon | senior_support_agent |
| Karthik Iyer | support_agent |
| Fatima Sheikh | support_agent |
| Arjun Mehta | supervisor |
| Divya Nair | support_agent |

## ⚠️ Schema change: `last_ingested_chunk_ids` column

If you already ran `scripts/init_db.py` against Postgres before this
change, you need to add the new column manually (or drop/recreate) -- there
is no Alembic migration tool set up yet, so pick one:

**Option A -- add the column (keeps existing data):**
```bash
docker exec -it motor_insurance_postgres psql -U agent -d insurance_support \
    -c "ALTER TABLE policy_documents ADD COLUMN last_ingested_chunk_ids JSON NOT NULL DEFAULT '[]';"
```

**Option B -- wipe and recreate (fine at this stage, since it's all synthetic data):**
```bash
docker compose down -v && docker compose up -d postgres
uv run python scripts/init_db.py
uv run python scripts/seed_customers.py
```

## RAG ingestion (`knowledge_base/` -> Pinecone)

`integrations/policy_parser.py` splits a policy Markdown file into one chunk
per clause (`### Clause X-1.0: ...` / `#### Clause X-1.1: ...` headings),
attaching the nearest `## Section N: ...` heading as context. Requires a
YAML frontmatter block at the top of every file:

```markdown
---
product_type: motor_comprehensive
version: v3.2
title: Motor Comprehensive Insurance Policy
---
```

`integrations/rag.py` embeds each chunk (`text-embedding-3-small` via the
IITM-provided OpenAI-compatible endpoint) and upserts into the Pinecone
`policies` index, with metadata:

| Key | Example |
|---|---|
| `policy_doc_name` | "Motor Comprehensive Insurance Policy" |
| `policy_version` | "v3.2" |
| `policy_created` | ISO timestamp of this ingestion run |
| `product_type` | "motor_comprehensive" |
| `policy_document_id` | Postgres `policy_documents.id` (for joining back) |
| `clause_id` | "OD-4.2" |
| `section` | "What This Policy Covers" |
| `text` | the actual clause text (so retrieval doesn't need a second lookup) |

**Re-ingestion replaces, it doesn't accumulate.** Each document's vector IDs
(`{product_type}::{version}::{clause_id}`) are recorded in
`policy_documents.last_ingested_chunk_ids`. On re-ingest, that stale list is
deleted from Pinecone *before* the new chunks are upserted -- so if you
remove or rename a clause in the source file, the old vector doesn't linger
as an orphan. New files under `knowledge_base/*.md` are picked up
automatically (no registration step needed) since `ingest_all()` just globs
the directory.

**Run it two ways:**

CLI:
```bash
uv run python scripts/ingest.py
```

API (once `uv run uvicorn main:app --reload` is running):
```bash
curl -X POST http://localhost:8000/ingest
```

Both call the exact same `ingest_all()` function, so behaviour is identical.

## Running the API

```bash
uv run uvicorn main:app --reload
```
Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/ingest` | Re-scan `knowledge_base/*.md` into Pinecone |
| POST | `/tickets` | Create a ticket (`customer_id`, `ticket_text`) |
| POST | `/tickets/{id}/process` | Run the agent graph on it |
| GET | `/tickets/{id}` | Fetch ticket + interaction history |

Create and process are deliberately separate steps -- a real ticket intake
(e.g. a customer submission form) and the AI's processing of it are
different events in time. Example flow:
```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 1, "ticket_text": "Does my policy cover a cracked windscreen?"}'
# -> {"ticket_id": 1, ...}

curl -X POST http://localhost:8000/tickets/1/process
# -> {"category": "coverage_question", "draft_response": "...", "escalated": false, ...}

curl http://localhost:8000/tickets/1
# -> ticket detail + interaction history
```

## The agent graph (`graph/`)

One agent, one LangGraph `StateGraph`, 3 separate LLM calls per ticket
(kept separate rather than merged so each is independently traceable in
LangSmith):

```
classify_ticket -> fetch_customer_context -> agent_reasoning -> summarize_and_draft
    -> faithfulness_check -> [escalation_gate] -> escalate_to_agent | present_to_human
```

**`fetch_customer_context` runs unconditionally for every ticket** (deterministic,
no LLM) -- it fetches the customer's name and policy/history context before
the LLM does anything. This guarantees a repeat customer's history is
actually available to the agent, rather than depending on the LLM happening
to decide to call `customer_history_lookup` itself. The LLM can still call
that tool again during `agent_reasoning` if it wants to refresh or
double-check something specific -- in that case its result overwrites the
baseline; otherwise the baseline from `fetch_customer_context` is left
untouched (the node deliberately omits the `customer_context` key from its
return value rather than returning `None`, since a `None` would clobber the
baseline when LangGraph merges the partial state update).

**`customer_id` is never trusted from the LLM.** If the LLM decides to call
`customer_history_lookup` itself during `agent_reasoning` (on top of the
guaranteed `fetch_customer_context` baseline), its tool-call arguments are
never used as-is for `customer_id` -- the real value from state always
overrides whatever the model supplied. This was a real bug found via manual
testing: nothing in the original prompt told the LLM the actual
`customer_id`, so when it decided to call the tool on its own initiative it
had to guess a number, which returned a fabricated "customer not found"
result for an actual, active customer. `customer_id` is a fact about which
ticket is being processed, not a parameter the model should control -- same
reasoning as why `escalate_to_agent` isn't an LLM-callable tool at all.

**Tools bound to the LLM (`graph/tools.py`):** only `policy_lookup`
(Pinecone RAG) and `customer_history_lookup` (the memory query over
`tickets`/`interactions`/`customer_policies`). The LLM freely decides which
to call, both, or neither, based on the ticket -- this is genuine
tool-selection, not a fixed if/else on category.

**`escalate_to_agent` is deliberately NOT an LLM tool.** Escalation is a
rule-based function (`make_escalation_gate` + `make_escalate_node`, both
driven by a single `_evaluate_escalation()` source of truth) that runs
*after* the LLM's turn, never something the model can trigger on its own
initiative. Rules, in order: faithfulness failure, the LLM's soft signal,
complaint category, coverage question with no active policy, more than
`MAX_TICKETS_PER_30_DAYS` tickets from the same customer in a rolling
window, and a claimed amount exceeding `HIGH_VALUE_CLAIM_THRESHOLD`. All
four thresholds/names below are **configured via `.env`, not hardcoded** --
see `_evaluate_escalation()`'s docstring in `graph/nodes.py` for the full
rule list and priority assignment (faithfulness failure and high-value
claim both get `HIGH` priority + the named supervisor; everything else
`MEDIUM`).

| Setting | Default | Purpose |
|---|---|---|
| `HIGH_VALUE_CLAIM_THRESHOLD` | 100000 | Claims above this (INR) always escalate |
| `MAX_TICKETS_PER_30_DAYS` | 2 | More than this many tickets in the window escalates |
| `REPEAT_TICKET_WINDOW_DAYS` | 30 | Rolling window used to count recent tickets |
| `SUPERVISOR_AGENT_NAME` | Arjun Mehta | Seeded agent assigned to HIGH priority escalations |

Every node factory that needs one of these (`make_escalation_gate`,
`make_escalate_node`, `make_fetch_customer_context_node`) reads it from
`settings` by default, but accepts an explicit override argument too --
that's what makes `tests/test_agent_nodes.py`'s boundary-value tests (e.g.
"exactly 2 tickets should NOT escalate") independent of whatever happens to
be in `.env` when the tests run.

**`faithfulness_check` is deterministic, not an LLM judge.** It checks that
every `clause_id` the draft claims to rely on actually appears in what
`policy_lookup` retrieved. A citation to a clause that was never retrieved
fails the check and routes to escalation -- this is the concrete
enforcement of "must not fabricate policies."

**Draft formatting is deterministic, not left to the LLM.** The LLM only
produces the substantive `draft_body` (no greeting, no sign-off); Python
code (`_format_customer_letter()`) wraps it with `Dear <customer name>,`
(falling back to `Dear Customer,` if the name is somehow missing) and a
generic `Warm regards, Motor Insurance Support Team` sign-off. The
sign-off deliberately doesn't name a specific human agent, since at draft
time it isn't known who will actually review and send it.

**Everything is testable without real API keys.** `llm`, `embed_fn`, and
`index` are all injectable into `build_graph()` -- tests use fake doubles
(see `tests/test_agent_graph.py`, `test_agent_nodes.py`, `test_agent_tools.py`).
Production code just calls `build_graph(session)` with no extra args.

**Running it for real** (once you have a ticket in the DB):
```python
from customer_support_agent.core import get_session
from customer_support_agent.graph import build_graph

with get_session() as session:
    graph = build_graph(session)
    result = graph.invoke({
        "ticket_id": 1,
        "customer_id": 1,
        "ticket_text": "Does my policy cover a cracked windscreen?",
    })
    print(result["draft_response"], result["escalated"])
```

## LangSmith tracing

Set in `.env`:
```
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=motor-insurance-support-agent
LANGSMITH_TRACING=true
```
`core/tracing.py` -> `configure_langsmith()` translates these into the env
vars LangChain/LangGraph read automatically (`LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) -- called once at graph build
time, no separate client object needed. With tracing on, every node's LLM
call, every tool call and its arguments, and the full graph run for a
ticket show up as a trace in the LangSmith UI -- useful evidence for your
Phase 9 Evaluation Report and Demo Script screenshots.

## Streamlit UI (`app.py` + `ui/`)

Two views in one app, gated by a simple login:

- **Customer Portal** -- submit a ticket (chat-style input), view past ticket
  history. The customer never sees the AI's draft directly, only the final
  human-approved response once a ticket is closed -- that's the
  human-in-the-loop boundary, not an oversight.
- **Agent Console** -- queue of tickets (escalated ones surfaced first),
  review the AI's summary + draft, edit if needed, then Approve & Send
  (closes the ticket) or Re-run AI.

`ui/` is kept separate from `customer_support_agent/` and talks to the
backend **over HTTP** (via `ui/api_client.py`), not by importing it
directly -- this is what makes the earlier API-layer work actually mean
something (UI and backend are properly decoupled, as they'd be in a real
deployment).

**Login is intentionally simplified**: everyone shares one password
(`DEMO_SHARED_PASSWORD` in `.env`, default `password1`). Customers additionally
enter an existing customer ID (validated against the DB via `GET
/customers/{id}`); agents pick their name from the 5 seeded agents. This is
explicitly **not production authentication** -- worth stating plainly in
your Engineering Justification rather than pretending otherwise.

### Running it

Two processes, both need to be up:

```bash
# Terminal 1
uv run uvicorn main:app --reload

# Terminal 2
uv run streamlit run app.py
```

Then open the local URL Streamlit prints (`http://localhost:8501`).

### Sharing a public URL (ngrok)

Streamlit's own local/network URL only works on your LAN. For a link you
can send someone outside it, without touching your local Docker Postgres:

```bash
# after streamlit is running on 8501
ngrok http 8501
```

`ngrok` prints a public `https://....ngrok-free.app` URL that tunnels to
your local Streamlit process. It's temporary -- live only while both your
machine and the tunnel are running -- but needs zero changes to your
database or deployment setup. (If you don't have ngrok installed:
`https://ngrok.com/download`, free account required for the URL.)
