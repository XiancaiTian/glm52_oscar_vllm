#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-192.168.14.129:80/ae/vllm_openai_glm52:v0.19.0-v2-stable-2p1d-usagefix-20260625_114616}"
EXPECTED_BASE_IMAGE_ID="${EXPECTED_BASE_IMAGE_ID:-sha256:d6faf4d3a5f7f3800a745f8aea15884c881ea20b1100993d83ca8c4f985bd7a5}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-glm52-oscar-vllm:phase0-baseline}"
RUNTIME_SOURCE_COMMIT="${RUNTIME_SOURCE_COMMIT:-a5a2ddfc3fb1b221a6eb41023b254ac54a98bd2c}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v docker >/dev/null

actual_base_image_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE}")"
if [[ "${actual_base_image_id}" != "${EXPECTED_BASE_IMAGE_ID}" ]]; then
  echo "base image ID mismatch: expected ${EXPECTED_BASE_IMAGE_ID}, got ${actual_base_image_id}" >&2
  exit 1
fi

docker build \
  --pull=false \
  --file "${REPO_ROOT}/docker/Dockerfile.glm52-baseline" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "RUNTIME_SOURCE_COMMIT=${RUNTIME_SOURCE_COMMIT}" \
  --tag "${OUTPUT_IMAGE}" \
  "${REPO_ROOT}"

docker image inspect \
  --format 'id={{.Id}} repo_digests={{json .RepoDigests}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "${OUTPUT_IMAGE}"
