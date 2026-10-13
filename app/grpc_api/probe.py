import argparse
import json

import grpc

from app.grpc_api.generated import nia_pb2, nia_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--plaintext", action="store_true")
    parser.add_argument("--question")
    args = parser.parse_args()
    channel = (grpc.insecure_channel(args.target) if args.plaintext else
               grpc.secure_channel(args.target, grpc.ssl_channel_credentials()))
    with channel:
        client = nia_pb2_grpc.AssistantStub(channel)
        reply = client.Health(nia_pb2.HealthRequest(), timeout=10)
        if reply.status != "ok":
            raise RuntimeError("NIA gRPC health failed")
        if args.question:
            result = client.Query(nia_pb2.QueryRequest(question=args.question, selected_file_id=49), timeout=30)
            if result.status != "answered":
                raise RuntimeError(f"NIA query failed: {result.status}")
            print(json.dumps({"status": result.status, "answer": result.answer}))
        else:
            print("NIA gRPC health: ok")


if __name__ == "__main__":
    main()
