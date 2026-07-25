# GLM‑5.2 baseline 恢复说明

本目录记录阶段 0 已落地的源码和镜像一致性信息。数据来自
`2026-07-24` 对已验证 Docker archive 的实际读取与 SHA256 核验。

## 恢复基线

- 原始源码 commit：
  `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`；
- 原始源码 tree：
  `9b1b1bf2a2f20459a1de4c0dc2ba73f23f6c2923`；
- 镜像 runtime patch commit：
  `a5a2ddfc3fb1b221a6eb41023b254ac54a98bd2c`；
- baseline 构建源码 commit：
  `fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd`；
- baseline 构建源码 tree：
  `31faa41bfb082983fa853dc1c499b48db3a81d3d`；
- 已验证镜像 ID：
  `sha256:d6faf4d3a5f7f3800a745f8aea15884c881ea20b1100993d83ca8c4f985bd7a5`；
- 镜像 archive SHA256：
  `b58fc5cf7874da92f5a97306d47c9d7d507332515a73e6812a7eb6145e9bed23`。

`runtime_source.sha256` 固定镜像相对原始 commit 的 4 个源码 patch。
`native_extensions.sha256` 固定镜像内 6 个 vLLM 原生扩展和 1 个 sparse MLA
splitmerge 扩展。

## 重建步骤

当前 Kubernetes 任务容器没有 Docker daemon，但可用 `skopeo`、`umoci`、
GNU tar、gzip 和 `uv`。2026-07-24 实际执行的无 daemon 重建命令为：

```bash
skopeo copy \
  docker-archive:/path/to/vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar \
  oci:/path/to/phase0-oci:verified

BASE_LAYOUT=/path/to/phase0-oci \
OUTPUT_LAYOUT=/path/to/phase0-candidate-oci \
./recovery/build_oci_baseline.sh
```

脚本拒绝 base manifest digest 不匹配的输入，从固定 Git commit 导出完整
index，生成确定性的标准 tar/gzip 层，并更新 OCI config、manifest 和 index。

在有 Docker daemon 的构建机上，也可以先校验并加载已验证 archive：

```bash
sha256sum -c \
  vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar.sha256
docker load -i \
  vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar
```

然后在本仓库根目录运行：

```bash
./recovery/build_baseline_image.sh
```

脚本会先拒绝 image ID 不匹配的 base image，再执行 `--pull=false` 构建。
Dockerfile 在构建阶段复核源码和原生扩展 SHA256。

## 当前验证状态

已完成：

- 34.75GB 镜像 archive 全量 SHA256；
- 原始 commit/tree 和完整历史恢复；
- 4 个 runtime patch 与镜像逐字节比对；
- 7 个原生扩展的 SHA256 冻结；
- Python 3.12 `py_compile`；
- `git diff --check`。

实际候选结果：

- OCI manifest：
  `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d`；
- OCI config：
  `sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d`；
- 新增 source layer：
  `sha256:352d47f649171770e32edb7e1112e8a31f6a5aead0f6160a30ad3a8eec6659c3`，
  33,253,726 字节；
- source layer diff ID：
  `sha256:e9287018ae318195cd2bdc8fdd88b5c6fd7f85c2c44db453d891c075f538ccf8`；
- 31/31 个基础层 descriptor 与已验证 OCI 完全相同；
- source layer 共 5,256 个 tar 成员和 4,711 个文件，内容与权限逐一匹配；
- 独立第二次生成得到相同 layer digest、diff ID 和大小；
- source layer 不含 `.so` 或 whiteout，因此 7 个已冻结原生扩展保持不变；
- 在已展开 rootfs 上复核 4 个 runtime source 和 7 个 native extension，
  实际结果为 11/11 `OK`。

未完成：

- 新 GitHub 远端尚未创建或确认；
- A800 TP=8 baseline 属于阶段 1，尚未开始。

额外执行的完整 rootfs 解包在约 72 分钟后终止。文件内容已经展开，4 个 runtime
source 和 7 个 native extension 均通过 SHA256；但 `umoci unpack` 因 NFS
xattr/元数据扫描耗时过长而没有得到完成状态，因此“unpack 命令完成”不得记为
通过。候选内容判定采用已验证 base archive、31/31 基础层 descriptor 一致性、
完整 source layer 遍历以及展开 rootfs 的 11/11 SHA256 证据。

定向 ruff 0.14.0 检查发现镜像历史 patch 的 3 个 E501 和 1 个 SIM102。
为保持恢复基线与已验证镜像逐字节一致，本阶段没有修改这些历史行。
