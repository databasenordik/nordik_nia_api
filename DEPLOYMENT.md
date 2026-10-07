# Initial NIA deployment

Project: `planar-ray-472112-e8`; region: `us-west1`.
Cloud Run service and Artifact Registry repository: `nordik-nia-api`.
Image path: `us-west1-docker.pkg.dev/planar-ray-472112-e8/nordik-nia-api/nordik-nia-api`.

This stage deploys the Python API with public ingress and unauthenticated Cloud Run
access. Integration with nordikdriveapi, application authentication changes,
Keycloak, VPC restrictions, gRPC, and mTLS are separate later steps.

## Automatic tests and deployment

Like nordikdriveapi, this repository uses GitHub Actions rather than a separate
Cloud Build webhook. Pull requests to `main` run CI. Pushes to `main` run the same
CI, then build a container, push it to Artifact Registry, and deploy Cloud Run.
Manual runs are also available in the Actions tab. Deployment waits for tests,
checks revision readiness, and requests the public `/health` endpoint.

CI runs the configuration tests and builds and starts the actual deployment image
on port 8080. Its container smoke test uses development mode without a database.
Images use unique commit/run tags. Deployment keeps Cloud Run environment values
and secret bindings intact; set them in the Cloud Run console.

GitHub Actions secrets:

| Secret | Value |
| --- | --- |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/724838782318/locations/global/workloadIdentityPools/github-pool-api/providers/github-provider-nia-api` |
| `GCP_SERVICE_ACCOUNT_EMAIL` | `github-nia-deployer@planar-ray-472112-e8.iam.gserviceaccount.com` |

The dedicated workload identity provider trusts only
`databasenordik/nordik_nia_api` on `refs/heads/main`. The deployment account has
Artifact Registry Writer on the NIA repository, Cloud Run Admin on the NIA service,
and Service Account User on its existing runtime identity. No Google JSON key or
VPC GitHub variables are needed for this initial public deployment.

## Initial Cloud Run environment

Only `APP_ENV=development` is needed to let the process start before its database
and credentials are provisioned. Cloud Run provides `PORT`; do not manually add
that reserved variable. The container listens on port 8080 in Cloud Run.

`GET /health` confirms the process runs. It can report reasoning as unavailable
until the provider is configured. `/ready` and chat requests need actual dependency
configuration. A healthy process alone does not mean the chat integration works.

Deployment uses one minimum instance, three maximum instances, one CPU, 1 GiB
memory, and a 300-second request timeout. One minimum instance avoids routine
scale-to-zero but is not an availability guarantee. HTTP startup and liveness
probes can use `/health`; `/ready` also checks voice dependencies and should not be
a liveness probe.

## Environment variables for the next stage

All fields in `app/config.py` accept uppercase environment names. Cloud Run values
override `.env` values; passwords and keys can be bound from Secret Manager.
The settings already support environment overrides. The duplicate hardcoded TTS
endpoint was removed so the speech client uses `XAI_TTS_URL` from settings.

| Variable | Purpose |
| --- | --- |
| `ASSISTANT_DATABASE_URL` | PostgreSQL URL for the isolated `assistant_runtime` role |
| `REASONING_PROVIDER` | `http` for an HTTP-compatible model provider |
| `REASONING_BASE_URL` | Actual reachable model API endpoint |
| `REASONING_MODEL` | Model served by that endpoint |
| `XAI_API_KEY` | Credential required by the current reasoning code, including the HTTP adapter |
| `PLANNER_MODE` | `ai` or `legacy` |
| `CORS_ORIGINS` | Explicit allowed browser origins, comma separated, if browser access is needed |
| `JWT_SECRET` | Session-signing secret; at least 32 characters in production |
| `AUTH_ADAPTER` | Current code supports `standalone` or `website`; Cloud Run public access does not bypass application auth |
| `STANDALONE_DEMO_USERNAME`, `STANDALONE_DEMO_PASSWORD`, `STANDALONE_DEMO_PRINCIPAL_ID` | Current standalone login credentials and authorized database principal UUID |
| `WEBSITE_JWT_SECRET`, `WEBSITE_JWT_ISSUER`, `WEBSITE_JWT_AUDIENCE` | Alternative website adapter settings |
| `LIVEKIT_URL`, `LIVEKIT_HEALTH_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Voice room service |
| `WHISPER_SERVICE_URL` | Speech recognition service |
| `TTS_PROVIDER`, `XAI_TTS_URL`, `KOKORO_URL` | Speech provider and endpoint |
| `DB_POOL_MIN`, `DB_POOL_MAX` | Database pool limits; defaults 2 and 12 per instance |

Tuple settings such as `REASONING_MODEL_CHOICES` use JSON arrays, for example
`["your-model"]`. The serving process does not automatically migrate the database.
Run migrations separately when configuring the database using the README's steps.

Before switching to `APP_ENV=production`, configure its required credentials,
`AUTH_COOKIE_SECURE=true`, database connectivity and privilege isolation, and
`LIVEKIT_API_SECRET` (currently required even if voice is unused).
Use `APPLY_APP_STUBS=false` when migrating an existing research database.

Check deployment logs in GitHub Actions and revision logs in Cloud Run if a build
or startup fails. Roll back by routing traffic to a previously healthy revision.
Public access is the explicitly selected initial configuration; network and
identity hardening will follow later.
