// Starts the API on the transport NIA_TRANSPORT selects, as the Python API's entrypoint does.
import { NiaConfig } from "./config"
import { Conversations } from "./conversations"
import { NiaEngine } from "./engine"
import { NiaGrpc } from "./grpc"
import { NiaHttp } from "./http"
import { HttpSession } from "./http-session"

const config = NiaConfig.load(process.env)
for (const warning of config.warnings) console.warn(warning)
// Browser sign-in settings are checked before anything connects. gRPC callers are trusted services
// that sign in to nothing, so the gRPC transport needs none of them.
const auth = config.transport === "http" ? HttpSession.load(process.env) : undefined
// Loopback by default; hosted deployments set APP_HOST=0.0.0.0. The platform's PORT wins over APP_PORT.
const hostname = process.env.APP_HOST ?? "127.0.0.1"
const port = Number(process.env.PORT ?? process.env.APP_PORT ?? 8000)
const engine = await NiaEngine.start(config)

if (auth) {
  NiaHttp.serve({ config, auth, engine, hostname, port })
  console.log(`NIA HTTP listening on ${hostname}:${port}`)
} else {
  const store = await Conversations.postgres(config.database)
  const server = await NiaGrpc.serve({
    store,
    engine: NiaGrpc.engine(engine.host),
    principal: config.principal,
    model: config.model.slice(config.model.indexOf("/") + 1),
    hostname,
    port,
  })
  console.log(`NIA gRPC listening on ${hostname}:${server.port}`)
  const stop = () =>
    void server
      .close()
      .then(() => Promise.all([store.close(), engine.host.close()]))
      .finally(() => process.exit(0))
  process.once("SIGTERM", stop)
  process.once("SIGINT", stop)
}
