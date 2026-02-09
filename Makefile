GRPC_SERVICE_PROTO_FILE=https://raw.githubusercontent.com/triton-inference-server/common/refs/heads/main/protobuf/grpc_service.proto
MODEL_CONFIG_PROTO_FILE=https://raw.githubusercontent.com/triton-inference-server/common/refs/heads/main/protobuf/model_config.proto
PROTO_DIR=./protos

.PHONY: run fetch-protos

run:
	uv run python3 main.py

fetch-protos:
	mkdir -p $(PROTO_DIR)
	curl -o $(PROTO_DIR)/grpc_service.proto $(GRPC_SERVICE_PROTO_FILE)
	curl -o $(PROTO_DIR)/model_config.proto $(MODEL_CONFIG_PROTO_FILE)
