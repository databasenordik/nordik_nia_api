# Deploying the Nia research service

This branch holds the TypeScript research engine (`backend/src/http.ts`). It deploys as its own Cloud Run service, `nordik-nia-research`, next to the production API `nordik-nia-api`. Nothing here changes `nordik-nia-api`, `cd-python.yml`, the gRPC contract, or the `assistant` / `assistant_api` schemas.

| | `nordik-nia-api` (main) | `nordik-nia-research` (this branch) |
|---|---|---|
| Deployed by | push to `main` (`cd-python.yml`) | push of a `nia-research-v*` tag (`cd-ts.yml`) |
| Image | `nordik-nia-api/nordik-nia-api` | `nordik-nia-api/nordik-nia-research` |
| Database role | `assistant_runtime` | `nia_research`, member of `opencode_research_ro` |
| Reads through | `assistant_api` | `research_api`, which wraps `assistant_api` |
| Callers | Go adapter (gRPC), website, voice | Next.js frontend only (HTTP) |

The steps run in order. Steps 1 to 4 are one-time.

## 1. Read the live API's configuration

Copy the connection settings from the running service instead of guessing them. The commands below print names and secret references only, never values.

```sh
P=planar-ray-472112-e8 R=us-west1
gcloud run services describe nordik-nia-api --project $P --region $R --format=json > nia-api.json
jq '.spec.template.metadata.annotations | with_entries(select(.key | test("cloudsql|vpc|network")))' nia-api.json
jq -r '.spec.template.spec.containers[0].env[] | "\(.name)\t\(.valueFrom.secretKeyRef.name // "-")"' nia-api.json
jq -r '.spec.template.spec.serviceAccountName' nia-api.json
```

Record:

- **Transport.** `NIA_TRANSPORT`, `http` when unset. The Go adapter keeps using `nordik-nia-api` either way.
- **Database path.** A `run.googleapis.com/cloudsql-instances` annotation means a Unix socket. A VPC connector or network interface means a private IP. Neither means a public IP. See step 5.
- **Principal.** The value behind `STANDALONE_DEMO_PRINCIPAL_ID` (or `NIA_GRPC_PRINCIPAL_ID`). Reuse it as `NIA_PRINCIPAL` so both services see exactly the same lists.

Check that the deploy identity can authenticate from a tag. The provider name is the `GCP_WORKLOAD_IDENTITY_PROVIDER` repository secret.

```sh
gcloud iam workload-identity-pools providers describe "$PROVIDER" --format='value(attributeCondition)'
```

If the condition pins `assertion.ref` to `refs/heads/main`, extend it to `refs/tags/nia-research-v*`. Otherwise `cd-ts.yml` fails at the `auth` step.

## 2. Database (DBA)

`backend/database/gateway.sql` adds things only:

- the `research_api` schema;
- the `opencode_research_ro` role, which is read-only and NOINHERIT;
- three `SECURITY DEFINER` functions over existing `assistant_api` objects.

It does not alter or grant on `assistant`, `assistant_api` or `public`. Every object it reads is created by `main`'s migrations, so the database must be at `main`'s latest migration first.

Run it as a role that is a member of `assistant_gateway_owner`, because the schema and functions are owned by that role:

```sh
psql "$ADMIN_DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/database/gateway.sql
```

Then create the service's login role. It gets nothing except the research role:

```sql
CREATE ROLE nia_research LOGIN PASSWORD '<generated>' NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
GRANT opencode_research_ro TO nia_research;
```

Verify it as `nia_research`:

```sql
SELECT research_api.identity();
SELECT research_api.catalog('<NIA_PRINCIPAL>', 49, false) ->> 'label';
```

On every connection the service also refuses to run if `nia_research` can do any of the following:

- read a table directly;
- create objects in `public`, `assistant`, `assistant_api` or `research_api`;
- bypass RLS.

So if the role has wider grants than above, the service fails at startup rather than running with them.

**Rollback** affects only this service:

```sql
DROP ROLE nia_research;
DROP SCHEMA research_api CASCADE;
DROP ROLE opencode_research_ro;
```

## 3. Secrets and runtime identity

Create these Secret Manager secrets:

| Secret | Holds |
|---|---|
| `nia-research-database-url` | `postgres://nia_research:<password>@<host>/<database>`; the host form depends on step 5 |
| `nia-research-api-key` | Key for the model endpoint. It can be the same xAI key the API uses. |
| `nia-research-jwt-secret` | 32+ random characters. **Not** the API's `JWT_SECRET`. |
| `nia-research-password` | Password for the primary sign-in |

Create a runtime service account, e.g. `nia-research@planar-ray-472112-e8.iam.gserviceaccount.com`. Grant it:

- `roles/secretmanager.secretAccessor` on those four secrets;
- `roles/cloudsql.client` if the database is reached through Cloud SQL.

The deploy identity (`GCP_SERVICE_ACCOUNT_EMAIL`) needs:

- `roles/iam.serviceAccountUser` on that runtime account;
- `run.services.update` on the new service. This is already the case if its Cloud Run role is project-wide.

## 4. Bootstrap the service

Create the service once with a placeholder image. The workflow then only swaps the image, so environment, secrets and connections stay as set here.

```sh
gcloud run deploy nordik-nia-research --project $P --region $R \
  --image us-docker.pkg.dev/cloudrun/container/hello \
  --service-account nia-research@$P.iam.gserviceaccount.com \
  --ingress all --allow-unauthenticated \
  --set-env-vars APP_ENV=production,NIA_MODEL=xai/<model>,NIA_BASE_URL=https://api.x.ai/v1,NIA_PRINCIPAL=<principal>,NIA_PRIVATE=false,NIA_SSL=<see step 5>,CORS_ORIGINS=<frontend origin> \
  --set-secrets NIA_DATABASE_URL=nia-research-database-url:latest,NIA_API_KEY=nia-research-api-key:latest,JWT_SECRET=nia-research-jwt-secret:latest,STANDALONE_DEMO_PASSWORD=nia-research-password:latest
  # plus the same --add-cloudsql-instances or --vpc-connector the API uses (step 1)
```

The service must accept unauthenticated requests, like the API. The frontend reaches it through a server-side rewrite that sends no Google identity, and access is enforced by the service's own sign-in.

### Environment mapping

| `nordik-nia-api` | `nordik-nia-research` | Note |
|---|---|---|
| `APP_ENV=production` | `APP_ENV=production` | Enables the startup checks |
| `ASSISTANT_DATABASE_URL` | `NIA_DATABASE_URL` | Different role: `nia_research`, never `assistant_runtime` |
| `REASONING_BASE_URL` | `NIA_BASE_URL` | |
| `REASONING_MODEL` | `NIA_MODEL` | Prefixed with a provider id, e.g. `xai/grok-4.20-0309-non-reasoning` |
| `XAI_API_KEY` | `NIA_API_KEY` | |
| `JWT_SECRET`, `JWT_TTL_SECONDS` | same names | Different secret value |
| `STANDALONE_DEMO_USERNAME`, `STANDALONE_DEMO_PASSWORD` | same names | Username defaults to `researcher` |
| `STANDALONE_DEMO_PRINCIPAL_ID` | `NIA_PRINCIPAL` | Same value |
| `CORS_ORIGINS` | `CORS_ORIGINS` (and/or `FRONTEND_ORIGIN`) | `*` is refused |
| `AUTH_COOKIE_SECURE=true` | always Secure in production | |
| `PORT` | `PORT` | Takes precedence over `APP_PORT` in both |
| none | `NIA_SSL`, `NIA_PRIVATE`, `NIA_REASONING_EFFORT`, `PUBLIC_TEST_PASSWORD` | `PUBLIC_TEST_PASSWORD` unset means no second account |

These have no counterpart, so the features they configure stay on `nordik-nia-api`:

- `AUTH_ADAPTER=website` and `WEBSITE_JWT_*`
- `LIVEKIT_*`, `XAI_TTS_*`, `WHISPER_*` and `KOKORO_*`
- `NIA_TRANSPORT` and `NIA_GRPC_*`
- the rate-limit and retention settings

## 5. Database connection

The service only skips TLS for loopback hosts (`127.0.0.1`, `localhost`, `::1`). Anything else needs `NIA_SSL=verify-full`, with a server certificate that matches the host name and is trusted by the container. Which path works depends on step 1:

- **Unix socket (`cloudsql-instances`).** Not supported yet. The connection code needs a small change to pass the socket directory to Bun's SQL client (`path`). Make that change on this branch before the first release.
- **Private or public IP.** Connect to the instance's DNS name with `NIA_SSL=verify-full`. Mount the Cloud SQL server CA from a secret and point `NODE_EXTRA_CA_CERTS` at it.

## 6. Release

```sh
git tag nia-research-v0.1.0 && git push origin nia-research-v0.1.0
```

`cd-ts.yml` runs in this order:

1. Run CI.
2. Stop if the service has not been bootstrapped.
3. Build and push the image.
4. Deploy the image as one instance. The new revision only receives traffic after `/health` answers, which it does once the database and engine are up.
5. Check `/health` publicly.

Point a separate frontend deployment at the service with `BACKEND_ORIGIN=<service URL>`. Leave every existing frontend and the Go adapter on `nordik-nia-api`.

**Rollback:**

```sh
gcloud run services update-traffic nordik-nia-research --to-revisions=<previous revision>=100
```

Or delete the service. `nordik-nia-api` is unaffected either way.

## Known limits

- **Chats.** Kept on the single instance and lost when a revision replaces it. Sign-ins are signed cookies and survive.
- **Feedback.** Written to the service log as `REPORT` lines, not to `assistant.feedback`.
- **Missing features.** No gRPC, website sign-in, voice, `/ready`, rate limiting or JSON (non-streaming) query responses. Those remain `nordik-nia-api` features until the engine is wired in behind it.
