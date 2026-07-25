#!/usr/bin/env bash
set -euo pipefail

BASE_LAYOUT="${BASE_LAYOUT:?set BASE_LAYOUT to the verified OCI layout}"
OUTPUT_LAYOUT="${OUTPUT_LAYOUT:?set OUTPUT_LAYOUT to a new output OCI layout}"
BASE_TAG="${BASE_TAG:-verified}"
OUTPUT_TAG="${OUTPUT_TAG:-phase0-baseline}"
SOURCE_COMMIT="${SOURCE_COMMIT:-fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd}"
EXPECTED_BASE_MANIFEST="${EXPECTED_BASE_MANIFEST:-sha256:1b8e808e44e7ef50be0cc8ff874597c37e09a2a428f06092ab570e6079086415}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v git >/dev/null
command -v gzip >/dev/null
command -v sha256sum >/dev/null
command -v skopeo >/dev/null
command -v tar >/dev/null
command -v uv >/dev/null

if [[ -e "${OUTPUT_LAYOUT}" ]]; then
  echo "output OCI layout already exists: ${OUTPUT_LAYOUT}" >&2
  exit 1
fi

source_tree="$(git -C "${REPO_ROOT}" rev-parse "${SOURCE_COMMIT}^{tree}")"
source_epoch="$(git -C "${REPO_ROOT}" show -s --format=%ct "${SOURCE_COMMIT}")"
created_at="$(date --utc --date="@${source_epoch}" +%Y-%m-%dT%H:%M:%SZ)"

base_manifest="$(
  skopeo inspect --format '{{.Digest}}' \
    "oci:${BASE_LAYOUT}:${BASE_TAG}"
)"
if [[ "${base_manifest}" != "${EXPECTED_BASE_MANIFEST}" ]]; then
  echo "base manifest mismatch: expected ${EXPECTED_BASE_MANIFEST}, got ${base_manifest}" >&2
  exit 1
fi

work_dir="$(mktemp -d /tmp/glm52-oci-build.XXXXXX)"
trap 'rm -rf -- "${work_dir:?}"' EXIT
payload_dir="${work_dir}/rootfs/opt/vllm_glm52_v1"
mkdir -p "${payload_dir}"

export GIT_INDEX_FILE="${work_dir}/git-index"
git -C "${REPO_ROOT}" read-tree "${SOURCE_COMMIT}"
git -C "${REPO_ROOT}" checkout-index \
  --all \
  --force \
  --prefix="${payload_dir}/"
unset GIT_INDEX_FILE

payload_files="$(find "${payload_dir}" -type f | wc -l)"
if [[ "${payload_files}" -ne 4711 ]]; then
  echo "unexpected payload file count: ${payload_files}" >&2
  exit 1
fi

(
  cd "${payload_dir}"
  sha256sum -c recovery/runtime_source.sha256
)

layer_tar="${work_dir}/source-layer.tar"
layer_gzip="${work_dir}/source-layer.tar.gz"
tar \
  --sort=name \
  --mtime="@${source_epoch}" \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --format=pax \
  --pax-option=delete=atime,delete=ctime \
  -C "${work_dir}/rootfs" \
  -cf "${layer_tar}" \
  opt/vllm_glm52_v1
gzip -n -9 -c "${layer_tar}" > "${layer_gzip}"

tar -tf "${layer_gzip}" >/dev/null
layer_diff_id="sha256:$(sha256sum "${layer_tar}" | awk '{print $1}')"
layer_digest="sha256:$(sha256sum "${layer_gzip}" | awk '{print $1}')"
layer_size="$(stat -c %s "${layer_gzip}")"

skopeo copy \
  "oci:${BASE_LAYOUT}:${BASE_TAG}" \
  "oci:${OUTPUT_LAYOUT}:${OUTPUT_TAG}"

uv run --no-project --offline python - \
  "${OUTPUT_LAYOUT}" \
  "${OUTPUT_TAG}" \
  "${layer_gzip}" \
  "${layer_digest}" \
  "${layer_diff_id}" \
  "${layer_size}" \
  "${SOURCE_COMMIT}" \
  "${source_tree}" \
  "${created_at}" \
  "${base_manifest}" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

(
    output_layout,
    output_tag,
    layer_path,
    layer_digest,
    layer_diff_id,
    layer_size,
    source_commit,
    source_tree,
    created_at,
    base_manifest,
) = sys.argv[1:]

layout = pathlib.Path(output_layout)
blobs = layout / "blobs" / "sha256"


def read_blob(digest: str) -> dict:
    return json.loads((blobs / digest.removeprefix("sha256:")).read_text())


def write_blob(value: dict) -> tuple[str, int]:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (blobs / digest).write_bytes(raw)
    return f"sha256:{digest}", len(raw)


index_path = layout / "index.json"
index = json.loads(index_path.read_text())
descriptor = next(
    item
    for item in index["manifests"]
    if item.get("annotations", {}).get(
        "org.opencontainers.image.ref.name"
    )
    == output_tag
)
manifest = read_blob(descriptor["digest"])
config = read_blob(manifest["config"]["digest"])

config["created"] = created_at
labels = config.setdefault("config", {}).setdefault("Labels", {})
labels.update(
    {
        "ai.intellif.glm52.base-image-id": (
            "sha256:d6faf4d3a5f7f3800a745f8aea15884c881ea20b1100993d83ca8c4f985bd7a5"
        ),
        "ai.intellif.glm52.source-tree": source_tree,
        "org.opencontainers.image.base.digest": base_manifest,
        "org.opencontainers.image.created": created_at,
        "org.opencontainers.image.revision": source_commit,
        "org.opencontainers.image.title": "GLM-5.2 A800 baseline recovery",
    }
)
config.setdefault("rootfs", {}).setdefault("diff_ids", []).append(
    layer_diff_id
)
config.setdefault("history", []).append(
    {
        "created": created_at,
        "created_by": "COPY verified Git tree /opt/vllm_glm52_v1",
        "comment": "stage 0 reproducible baseline source layer",
    }
)

config_digest, config_size = write_blob(config)
manifest["config"] = {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": config_digest,
    "size": config_size,
}
manifest.setdefault("layers", []).append(
    {
        "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
        "digest": layer_digest,
        "size": int(layer_size),
    }
)
manifest.setdefault("annotations", {}).update(
    {
        "org.opencontainers.image.created": created_at,
        "org.opencontainers.image.revision": source_commit,
        "org.opencontainers.image.title": "GLM-5.2 A800 baseline recovery",
    }
)

layer_blob = blobs / layer_digest.removeprefix("sha256:")
shutil.copyfile(layer_path, layer_blob)
if hashlib.sha256(layer_blob.read_bytes()).hexdigest() != layer_digest.removeprefix(
    "sha256:"
):
    raise RuntimeError("copied layer digest mismatch")

manifest_digest, manifest_size = write_blob(manifest)
descriptor["digest"] = manifest_digest
descriptor["size"] = manifest_size
index_path.write_text(
    json.dumps(
        index,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
)

print(f"source_commit={source_commit}")
print(f"source_tree={source_tree}")
print(f"created_at={created_at}")
print(f"base_manifest={base_manifest}")
print(f"layer_diff_id={layer_diff_id}")
print(f"layer_digest={layer_digest}")
print(f"candidate_manifest={manifest_digest}")
PY

skopeo inspect \
  --format 'digest={{.Digest}} created={{.Created}} architecture={{.Architecture}} os={{.Os}} layers={{len .Layers}}' \
  "oci:${OUTPUT_LAYOUT}:${OUTPUT_TAG}"
