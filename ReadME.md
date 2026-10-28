# NIA API

Go chat adapter, feature flag, and gRPC setup: [docs/GRPC_INTEGRATION.md](docs/GRPC_INTEGRATION.md).

FastAPI service for the NIA research assistant. It plans a question against one selected list, reads the current authorized rows through `assistant_api`, and returns the answer. Text and voice use the same core.

The runtime connects as `assistant_runtime`. Research rows are read only through `assistant_api` functions. Production startup runs a privilege self-test and refuses to boot if that role can read website auth tables.

## Layout

```text
app/            assistant core
  api/          HTTP routes
  db/           pool, migrator, and SQL migrations
  planning/     AI planner and plan validation
  execution/    query execution and answers
  security/     session auth and access scope
benchmarks/     planner and release checks
Dockerfile      API image
```

## Requirements

Python 3.12 and PostgreSQL. The database needs a migrator role and an `assistant_runtime` role.

Voice also needs a LiveKit server, the local Whisper service, and either xAI speech or Kokoro. Those processes live outside this repository.

## Configure

Settings load from `.env` in this directory, then from `../.env`. Keep that file out of git.

`APP_ENV=development` still boots when Postgres is down. `APP_ENV=production` exits if the database is down or the privilege self-test fails.

```text
APP_ENV=development
ASSISTANT_DATABASE_URL=postgresql://assistant_runtime:...@localhost:5432/nia
ASSISTANT_MIGRATOR_DATABASE_URL=postgresql://assistant_migrator:...@localhost:5432/nia
ASSISTANT_RUNTIME_PASSWORD=
ASSISTANT_MIGRATOR_PASSWORD=
APPLY_APP_STUBS=true
AUTH_ADAPTER=standalone
JWT_SECRET=
STANDALONE_DEMO_USERNAME=
STANDALONE_DEMO_PASSWORD=
PLANNER_MODE=ai
REASONING_PROVIDER=http
REASONING_BASE_URL=
REASONING_MODEL=grok-4.20-0309-non-reasoning
XAI_API_KEY=
CORS_ORIGINS=http://localhost:3000
```

`JWT_SECRET` must be at least 32 characters when `APP_ENV=production`. Set `AUTH_COOKIE_SECURE=true` there as well.

`APPLY_APP_STUBS=false` is for an existing research database. The migrator then skips the local stub, seed, and demo-grant files: `003_app_stubs.sql`, `007_seed.sql`, `038_additional_potential_demo.sql`, and `042_regression_suite_grants.sql`.

`AUTH_ADAPTER=website` accepts a short-lived access token signed by the website. Set `WEBSITE_JWT_SECRET`, `WEBSITE_JWT_ISSUER`, and `WEBSITE_JWT_AUDIENCE`. The browser exchanges that token at `POST /api/auth/session` and then sends the `nia_session` cookie.

`PLANNER_MODE=ai` is the default. `PLANNER_MODE=legacy` uses the previous compiler. Restart the process after changing it.

## Install and run

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps .
python -m app.db.migrate
python -m app.db.privilege_selftest
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`uv sync` in this directory installs the same package for local development. The voice extra is separate:

```bash
uv sync --extra voice
```

The image listens on `$PORT`, or port 8000 when that variable is unset.

```bash
docker build -t nia-api .
docker run --env-file .env -p 8000:8000 nia-api
```

Pass `--build-arg INSTALL_VOICE=true` to install the voice lockfile in the image.

Retention and index maintenance is one shot:

```bash
python -m app.observability.worker
```

Start the voice worker after the voice extra is installed:

```bash
python -m app.realtime.worker start
```

## HTTP

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process is up. Dependencies are not checked. |
| GET | `/ready` | Postgres, Whisper, LiveKit, and reasoning configuration. `503` when a check fails. |
| POST | `/api/auth/login` | Standalone username and password. Sets the session cookie. |
| POST | `/api/auth/session` | Exchange a website access token for the session cookie. |
| GET | `/api/auth/session` | Current session. |
| GET | `/api/auth/me` | Current principal. |
| POST | `/api/auth/logout` | Clear the session cookie. |
| GET | `/api/assistant/datasets` | Lists this caller may open. |
| POST | `/api/assistant/query` | Ask a question on the selected list. |
| GET | `/api/assistant/conversations` | Conversations for the current principal. |
| GET | `/api/assistant/conversations/{id}` | One conversation. |
| PATCH | `/api/assistant/conversations/{id}` | Rename a conversation. |
| DELETE | `/api/assistant/conversations/{id}` | Delete a conversation. |
| POST | `/api/assistant/feedback` | Report an answer. |
| POST | `/api/livekit/token` | Short-lived LiveKit room token. |
| POST | `/api/voice/speech` | Synthesize speech for the current session. |

Login, query, LiveKit token, and speech routes are rate-limited inside the process. Put a shared gateway limiter in front when more than one replica is running.

## Data access

A conversation stays on one selected list. The registered lists are file 49 (student master list), 91 (confirmed deaths), and the grant-required lists 93 (additional deaths) and 94 (potential records). The default people list is file 49.

Queries run as `assistant_runtime` through `assistant_api`. `GET /health` is liveness only. `GET /ready` is the dependency check.


### New user/file chat-history schema upgrade

New migration: `app/db/migrations/074_user_file_conversations.sql`. It extends
`assistant.conversations` with `user_id` and a stored `file_id` alias, enforces one
conversation per user/file, and adds `exchange_id` plus message scope metadata to
`assistant.messages`. Both messages in a question/answer pair share the exchange ID.
No new tables are created.

For developers using migrations, apply the normal migration runner during rollout:

```powershell
python -m app.db.migrate
```

For a manual incremental upgrade of an already initialized database, execute
`nordikdriveapi/init.sql` instead. It contains the same new SQL in a transaction
and records migration 074 when the migration tracking table exists.
`nordikdriveapi/assistant.sql` is the full schema bundle and labels the new section
`BEGIN NEW CHANGES TO EXECUTE: USER/FILE CHAT HISTORY (MIGRATION 074)`.
Choose one upgrade route; the application code update does not run SQL itself.
Legacy rows without a verified application user mapping remain `user_id = NULL`.
