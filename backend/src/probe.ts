// Health check against a running gRPC service, like nordik_nia_api's app/grpc_api/probe.py:
//   bun run backend/src/probe.ts --target host:443 [--plaintext]
import { parseArgs } from "node:util"
import * as grpc from "@grpc/grpc-js"
import { NiaGrpc } from "./grpc"

const { values } = parseArgs({ options: { target: { type: "string" }, plaintext: { type: "boolean" } } })
if (!values.target) throw new Error("--target is required")
const client = new NiaGrpc.Assistant(
  values.target,
  values.plaintext ? grpc.credentials.createInsecure() : grpc.credentials.createSsl(),
) as unknown as grpc.Client & {
  Health(
    request: object,
    options: grpc.CallOptions,
    callback: (error: grpc.ServiceError | null, response: { status: string }) => void,
  ): void
}
const reply = await new Promise<{ status: string }>((resolve, reject) =>
  client.Health({}, { deadline: Date.now() + 10_000 }, (error, response) => (error ? reject(error) : resolve(response))),
).finally(() => client.close())
if (reply.status !== "ok") throw new Error("NIA gRPC health failed")
console.log("NIA gRPC health: ok")
