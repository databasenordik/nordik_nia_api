#!/usr/bin/env bash
set -euo pipefail
# Isolated development fixture ONLY. Never uses a live connection or an existing container.
dump=${1:?Usage: restore.sh /path/to/postgres-live-2026-10-15.dump /path/to/profile.json}
profile=${2:?Provide an output profile path outside the repository}
root=$(cd "$(dirname "$0")/../.." && pwd)
expected=98d3478575acddbb0e45b03914937229569109a5be4c4cd1e3ec9f72e2ea5e7e
test "$(sha256sum "$dump" | cut -d' ' -f1)" = "$expected" || { echo 'Dump fingerprint mismatch' >&2; exit 1; }
name=opencode-research-fixture
if docker container inspect "$name" >/dev/null 2>&1; then echo "Fixture container already exists: $name. Reuse it or explicitly remove your fixture before restoring." >&2; exit 1; fi
work=$(mktemp -d)
trap 'rm -f "$work/restore.sql" "$work/restore.log"; rmdir "$work"' EXIT
docker run -d --name "$name" -e POSTGRES_HOST_AUTH_METHOD=trust -p 127.0.0.1::5432 postgres:15 >/dev/null
for i in {1..60}; do
  if docker exec "$name" pg_isready -U postgres >/dev/null; then break; fi
  sleep 1
done
docker exec "$name" psql -X -U postgres -v ON_ERROR_STOP=1 -c 'CREATE ROLE assistant_gateway_owner NOLOGIN; CREATE ROLE assistant_runtime NOLOGIN; CREATE ROLE cloudsqlsuperuser NOLOGIN;' >/dev/null
pg_restore --file="$work/restore.sql" "$dump"
sed '/^SET transaction_timeout = 0;/d' "$work/restore.sql" | docker exec -i "$name" psql -X -U postgres -v ON_ERROR_STOP=1 > "$work/restore.log" 2>&1
docker exec -i "$name" psql -X -U postgres -v ON_ERROR_STOP=1 < "$root/backend/database/gateway.sql" >> "$work/restore.log" 2>&1
docker exec "$name" psql -X -U postgres -v ON_ERROR_STOP=1 -c "CREATE ROLE research_fixture LOGIN PASSWORD 'research-fixture-only' NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS; GRANT opencode_research_ro TO research_fixture; INSERT INTO assistant.access_grants(principal_id,file_id) VALUES ('research-fixture',93),('research-fixture',94);" >> "$work/restore.log" 2>&1
port=$(docker port "$name" 5432/tcp | sed 's/.*://')
umask 077
printf '{"url":"postgres://research_fixture:research-fixture-only@127.0.0.1:%s/postgres","principal":"research-fixture","private":true,"ssl":"disable"}\n' "$port" > "$profile"
echo "Created isolated PostgreSQL 15 fixture. Restricted development profile written to $profile"
echo "Cleanup: docker rm -f -v $name"
