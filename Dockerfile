FROM oven/bun:1.4.2-slim

WORKDIR /app
# Built from the repository root so the Bun workspace, its catalog versions and the dependency
# patches resolve exactly as bun.lock records them.
COPY package.json bun.lock bunfig.toml ./
COPY backend ./backend
# PGlite is a test-only dependency, but drizzle-orm names it as an optional peer, so Bun installs it
# even here. Nothing at runtime imports it.
RUN bun install --frozen-lockfile --production && rm -rf node_modules/@electric-sql

ENV NODE_ENV=production
# The API listens on loopback unless told otherwise; a container must accept outside connections.
ENV APP_HOST=0.0.0.0
# Cloud Run provides PORT; local containers default to 8080.
ENV PORT=8080
EXPOSE 8080
USER bun
# NIA_TRANSPORT picks gRPC (nordik.nia.v1.Assistant) or HTTP, as in the Python API's image.
CMD ["bun", "run", "backend/src/main.ts"]
