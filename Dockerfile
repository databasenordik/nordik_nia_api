FROM oven/bun:1.4.2-slim

WORKDIR /app
# Built from the repository root so the Bun workspace, its catalog versions and the dependency
# patches resolve exactly as bun.lock records them.
COPY package.json bun.lock bunfig.toml ./
COPY backend ./backend
RUN bun install --frozen-lockfile --production

ENV NODE_ENV=production
# The API listens on loopback unless told otherwise; a container must accept outside connections.
ENV APP_HOST=0.0.0.0
# Cloud Run provides PORT; local containers default to 8080.
ENV PORT=8080
EXPOSE 8080
USER bun
CMD ["bun", "run", "backend/src/http.ts"]
