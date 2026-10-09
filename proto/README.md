# Shared NIA gRPC contract

`nia.proto` is mirrored in `nordikdriveapi/proto/nia.proto`. Keep both copies equal.
Generated Python files live in `app/grpc_api/generated`; generated Go files live
in `nordikdriveapi/internal/nia/pb`. Generated files are checked in so builds do
not need protoc.

Generator versions: `grpcio-tools==1.78.0`, `protoc-gen-go@v1.36.11`, and
`protoc-gen-go-grpc@v1.5.1`. The Python generator is kept compatible with the
runtime protobuf 6.x pins. Generate Python from this repository root:

```sh
python -m grpc_tools.protoc -I proto --python_out=app/grpc_api/generated --grpc_python_out=app/grpc_api/generated nia.proto
```

Then change the generated `nia_pb2_grpc.py` import from `import nia_pb2 as
nia__pb2` to `from . import nia_pb2 as nia__pb2` for package-relative imports.
Generate Go from the Go repository root with these plugins on PATH:

```sh
protoc -I proto --go_out=. --go_opt=module=nordik-drive-api --go-grpc_out=. --go-grpc_opt=module=nordik-drive-api nia.proto
```

`nordik.nia.v1.Assistant/Query` is a unary text query. It carries a selected file
ID and optional community selection, model, and conversation ID. The response
contains status, answer, conversation ID, file ID, error details, and the existing
HTTP-style result serialized in `result_json`. `Health` reports process liveness.
Neither RPC currently requires credentials or authenticates caller identity.
