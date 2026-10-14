# ServerlessLLM 深入学习文档

- 研究对象：`ServerlessLLM/ServerlessLLM`，fork 为 `jasonchen505/ServerlessLLM`
- 核查时间：2026-10-14，基线为上游 `main` HEAD `9f50241b`（`fix: release pipeline (#309)`，2026-03-06）
- 研究方式：只读核查，未跑任何代码，README 中的性能数字未独立复现
- 本文档位置：fork 的 `notes/learning` 分支，`main` 不动，不进入上游 PR

## 一句话定位

ServerlessLLM 是 OSDI'24 论文《ServerlessLLM: Low-Latency Serverless Inference for Large Language Models》的开源实现（Fu 等）。
它要解决的是 serverless 场景下 LLM 推理的两个痛点：冷启动慢（加载几十 GB checkpoint 需要几十秒），以及
GPU 利用率低（一个模型独占一张卡，大量时间空闲）。解法是三件套：超快 checkpoint 加载、多模型共享 GPU 的存储感知调度、
推理与 LoRA 微调共享同一套资源。README 的宣传口径是“加载快 6-10 倍，10 个模型跑在 1 张 GPU 上”。

## Fork 状态与分支卫生警告

- Fork 的 `main` 比上游 `main` 超前 1 个提交、落后 0 个。超前的提交是 `c9acb36b`（`Add markdown files`，2026-06-27），
  仅新增 4 个个人笔记 md 文件（`INTERVIEW_DEEP_DIVE.md`、`INTERVIEW_PREPARATION.md`、`LEARNING_INCREMENTAL.md`、
  `REPRODUCTION_PLAN.md`，共约 3265 行新增、0 删除）。
- 本 `notes/learning` 分支建在 fork `main` 之上，因此也包含这 4 个个人笔记文件。这是个人学习分支，可以接受。
- 但未来任何上游贡献分支必须从上游 `main`（`9f50241b`）新建，绝不能从 fork `main` 建，否则 4 个个人笔记会进入 PR diff。
- 上游 `main` HEAD 停在 2026-03-06。2026-04/05 合并的 #328、#337 等目标分支是 `v1-beta` / `v1-batch`，不是 `main`。
  开发重心已不在 `main` 上，见文末“维护者活跃度与风险”。

## 解决什么问题：Serverless 推理为什么难

传统推理服务是“常驻式”的：模型常驻显存，请求来了直接算。Serverless 要求按需启停、闲时缩到零，带来三个代价：

1. **冷启动代价**：每次拉起实例都要从磁盘把 checkpoint 读进显存。SafeTensors 加载 Qwen3-32B 需要约 20 秒，
   请求延迟里加载占大头，serverless 的弹性优势被冷启动吃掉。
2. **GPU 碎片**：一个 8B 模型只占一张 H100 的一小部分，但独占整卡；多个小模型各占一卡，集群利用率低。
3. **切换代价**：GPU 上换一个模型，要卸载旧模型、加载新模型，切换过程本身就是一次冷启动。

ServerlessLLM 的回答是：把加载做快一个数量级（让切换便宜到可以接受），再用调度器把“哪个模型在哪个节点的显存/内存里”
当成一等公民来规划（让切换尽量命中已加载的状态），最后让微调用同一套资源（避免推理集群和微调集群各闲一半）。

## 架构与数据流

顶层结构（上游 `main`，共 324 个条目）：

- `sllm/`（37 文件）：控制面 + 调度 + 路由 + 推理/微调后端，纯 Python + Ray。
- `sllm_store/`（93 文件）：快速 checkpoint 加载器。Python 壳（gRPC client/server、transformers 集成）+ C++ 核心（`csrc/`）+ vLLM patch。
- `tests/`（24 文件）、`benchmarks/`（48）、`docs/`（40）、`examples/`（24）、`configs/`（2）。
- 根目录：`Dockerfile`、`entrypoint.sh`、`setup.py`（`serverless-llm` 0.8.0，入口 `sllm = sllm.cli.clic:cli`）、`pyproject.toml`。
- 注意：**没有 `k8s/`、`kubernetes/`、`helm/` 目录**。K8s 部署仍是未实现的功能请求（issue #120）。
  部署文档只有单机、多机、Slurm（`docs/deployment/`）。

请求数据流：

```
curl POST /v1/chat/completions (OpenAI 兼容, head 节点 :8343)
  -> sllm/app_lib.py:create_app() FastAPI (sllm start 启动)
    -> SllmController (sllm/controller.py, Ray actor)
      -> Router: RoundRobinRouter / MigrationRouter (sllm/routers/)
        -> auto_scaler(): 按并发数算期望实例数, min_instances 默认为 0, 即 scale-to-zero
        -> Scheduler: FcfsScheduler / StorageAwareScheduler (sllm/schedulers/)
          -> StoreManager (sllm/store_manager.py, Ray actor, 每节点一个 SllmLocalStore)
            -> SllmStoreClient gRPC -> sllm-store server (:8073, sllm_store/server.py:StorageServicer)
              -> C++ CheckpointStore: O_DIRECT 读盘 -> pinned memory pool -> cudaMemcpy 到 GPU
          -> 调度器按存储感知 / 迁移计划选节点, 分配 Ray 资源
        -> Worker 上 sllm/inference_instance.py:start_instance() 起 Ray actor
          -> 后端: VllmBackend / TransformersBackend / DummyBackend (sllm/backends/)
            -> vLLM 需打 sllm_store/vllm_patch/sllm_load.patch (注册 serverless_llm load format)
            -> 生成 token, 流式返回
  <- 响应经 Router -> FastAPI 返回

微调旁路: POST /v1/fine-tuning/jobs -> Controller._process_ft_jobs()
  -> PeftLoraBackend (sllm/ft_backends/peft_lora_backend.py) 独立 FT 实例 -> 产出 adapter
  -> Controller.register_lora_adapter() 注册 -> 推理侧按需加载 LoRA (TransformersBackend.load_lora_adapter)
```

关键设计判断：控制面（Controller / Router / Scheduler / StoreManager）全是 Ray actor，
数据面（checkpoint 字节流）走 sllm-store 的 gRPC + C++ 直通 GPU。控制与数据分离，
所以调度器能拿到每节点的 pinned pool 占用、IO 队列延迟等状态，再做存储感知决策。

## 三大创新点：论文主张 vs 代码落地

论文正文未逐页核对，以下对照以 README 主张 + 代码实测为准。

| 创新点 | README 主张 | 代码落地 |
|---|---|---|
| 超快 checkpoint 加载 | 自定义存储格式 + O_DIRECT I/O，比 SafeTensors 快 6-10 倍。表格：Qwen3-32B Random 20.6s → 3.2s（6.40 倍），Cached 12.5s → 1.3s（9.95 倍），H100 + NVMe SSD | 真实落地，核心就是 `sllm_store`。`model.cpp:96` 的 `open(..., O_DIRECT \| O_RDONLY)`、`ToHost(num_threads)` 多线程加载、`PinnedMemoryPool` 分块 pinned memory、`cudaMemcpy`、gRPC server/client、`checkpoint/` 写 `tensor.data_*` + `tensor_index.json`。边界：2025-12-14 提交 `b27e3bf4`（#318）支持“不用 O_DIRECT 读模型”（`/dev/shm` 等场景 O_DIRECT 受限，需旁路）。README 数字为项目方自测，未独立复现 |
| GPU 多路复用 | 多模型共享 GPU、快速切换、存储感知调度、按需 scale-to-zero、live migration；“10 个模型跑在 1 张 GPU 上” | 控制面完整落地。`StorageAwareScheduler`（加载时间估算 + 迁移计划）、`StoreManager/SllmLocalStore`（pinned pool LRU + IO 队列）、`RoundRobinRouter.auto_scaler()`（`min_instances=0` 即 scale-to-zero）、`MigrationRouter` + `VllmBackend.get_current_tokens()/resume_kv_cache()` + `TransformersBackend.resume_generate()`（KV 恢复式迁移）。`examples/live_migration/`、`examples/storage_aware_scheduling/` 有可跑示例。“10 模型 / 1 GPU”是宣传口径，仓库内未找到对应复现基准 |
| 推理 + LoRA 微调统一 | 推理与 LoRA 微调共享资源；一个 base 模型 + 数百 LoRA adapter 按需服务；微调完的 adapter 可立即用于推理 | 架构上已落地且已分离为独立后端。`PeftLoraBackend`（#251 从 transformers 后端分离，2025-10-14 合入）、`POST /v1/fine-tuning/jobs` + `FineTuningJobStore`、推理侧 `load_lora_adapter()` + `Controller.register_lora_adapter()`。文档：`docs/features/peft_lora_fine_tuning.md`、`docs/features/peft_lora_serving.md`。未核实：“数百 adapter 同时服务”的规模声明无仓库内基准支撑 |

## 核心机制深挖

### 1. 快加载：为什么能比 SafeTensors 快 6-10 倍

四层叠加，缺一不可：

- **自定义格式**：`sllm_store/csrc/checkpoint/checkpoint.cpp`（`tensor_writer.cpp`）把模型写成 `tensor.data_*` + `tensor_index.json`，
  按顺序读优化，而不是按 safetensors 的通用布局随机访问。
- **O_DIRECT**：`model.cpp:96` 用 `O_DIRECT | O_RDONLY` 读盘，绕过 OS page cache，避免 page cache 拷贝与污染，
  对几十 GB 的 checkpoint 来说，page cache 不是加速而是负担。
- **Pinned memory pool**：`pinned_memory_pool.cpp` 维护分块 pinned（页锁定）内存，DMA 可直接从 pinned memory 到 GPU，
  省掉 pageable memory 的 staging 拷贝。`SllmLocalStore._lru_eviction()` 做 LRU 淘汰。
- **多线程 + gRPC**：`Model.ToHost(num_threads)` 多线程并行读盘；Python 侧 `SllmStoreClient` 经 gRPC 调 `StorageServicer`
 （默认 `:8073`，`num_thread=4, chunk_size=32MB`，见 `sllm_store/server.py`），`MemCopyChunk` / `MemCopyHandle`
  描述 GPU 拷贝（`proto/storage.proto`）。

独立使用：`pip install serverless-llm-store` 后，`sllm-store save` 转换模型，`sllm-store start` 起 store server，
Python 里 `from sllm_store.transformers import load_model` 即可 6-10 倍速加载。这是整个项目最容易单独复用的部分，
也是它区别于纯调度器项目的关键。

### 2. 存储感知调度：调度器为什么要知道“模型在哪”

普通调度器只看 CPU/GPU 余量。ServerlessLLM 的 `StorageAwareScheduler(FcfsScheduler)`（`sllm/schedulers/storage_aware_scheduler.py`）
额外估算“把这个模型加载到这个节点要多久”：

- 输入：`SllmLocalStore.get_store_info()` 返回的 `disk_models`（磁盘上有哪些模型）、`pinned_memory_pool`（内存里已缓存哪些）、
  IO 队列延迟（前面排了多少加载任务）。
- 逻辑：`schedule()` 用 pinned pool 命中情况与磁盘带宽估算加载时间，优先选加载最快的节点；
  `get_migration_plans()` 产出 `MigrationPlan` / `MigrationPlans`，把热模型预先迁到有空闲的节点。
- 效果：调度决策从“找空 GPU”变成“找加载代价最小的节点”，冷启动在调度层就被最小化，而不是等加载时再硬扛。

`FcfsScheduler` 本身（`sllm/schedulers/fcfs_scheduler.py`）含 TODO（`consider other resources`、专用 worker 节点管理类未实现），
说明调度器的资源模型还不完整。

### 3. Scale-to-zero 与自动伸缩

`RoundRobinRouter.auto_scaler()`（`sllm/routers/roundrobin_router.py`）：`desired = ceil(request_count / target)`，
夹在 `min_instances..max_instances` 之间。默认配置（`sllm/cli/default_config.json`）是
`backend: vllm`，`auto_scaling_config: {metric: concurrency, target: 1, min_instances: 0, max_instances: 10, keep_alive: 0}`。
`min_instances=0` 即闲时缩到零，`keep_alive=0` 即无保留时间。伸缩循环是 `_auto_scaler_loop()`，
实例生命周期由 `_create_instance()` / `_start_instance()` / `_stop_instance()` / `_finish_instance()` 管理。

### 4. Live migration：不中断请求的迁移

`MigrationRouter(RoundRobinRouter)`（`sllm/routers/migration_router.py`）+ 后端配合：

- `execute_migration_plan()` 执行迁移计划，`get_instance_status()` 查实例状态。含 TODO（`Add max_queue_length to target_instance`）。
- vLLM 后端：`VllmBackend.get_current_tokens()` / `resume_kv_cache()`（`sllm/backends/vllm_backend.py`），
  把正在生成的 KV cache 状态带到目标实例，而不是丢弃重算。含 TODO（`stream results`）。
- Transformers 后端：`resume_generate()`，另有 2 处 FIXME（`max_tokens` 达上限的 corner case）。

迁移的本质是“KV cache 不丢”，否则迁移就退化成一次冷启动 + 重算。

### 5. LoRA 微调统一：为什么微调要和推理放一起

`PeftLoraBackend`（`sllm/ft_backends/peft_lora_backend.py`）：`fine_tuning()`（HF datasets + PEFT LoRA + Trainer 风格训练）、
`_load_dataset()`；`FineTuningStatus`（start/update_metrics/log/complete/fail/abort）。
它在 2025-10-14 由 #251 从 transformers 后端分离，说明微调逻辑一度与推理后端耦合，后来才拆开。

推理侧 `TransformersBackend.load_lora_adapter()` 按需加载 adapter，`Controller.register_lora_adapter()` 注册。
微调产出的 adapter 不需要重启服务即可用于推理，这是“统一”的实际含义：共享同一套 StoreManager / 调度 / 模型注册，
而不是两套独立集群。

## 核心模块速查

| 模块 | 文件 | 职责与关键符号 |
|---|---|---|
| 请求入口 | `sllm/app_lib.py` | `create_app()`；路由：`GET /health`、`POST /register\|/update\|/delete`、`POST /v1/chat/completions`（`generate_handler`）、`POST /v1/embeddings`、`POST /v1/fine-tuning/jobs`、`GET /v1/fine_tuning/jobs/{id}`、`POST .../cancel`、`GET /v1/models` |
| 控制器 | `sllm/controller.py` | `SllmController`：`start()`（按 `enable_storage_aware` 选调度器、按 `enable_migration` 选 Router）、`register(model_config)`、`register_ft_job()`、`_process_ft_jobs()` |
| 存储管理 | `sllm/store_manager.py` | `StoreManager`（`initialize_cluster()`、`register()`、`load_to_host()`、`register_lora_adapter()`、`download_transformers_model()`、`download_vllm_model()`）；`SllmLocalStore`：`loading_loop()`（IO 队列 + 估计完成时间）、`_lru_eviction()`、`get_store_info()` |
| 调度器 | `sllm/schedulers/fcfs_scheduler.py` | `FcfsScheduler`：`allocate_resource()` / `deallocate_resource()` / `_control_loop()` / `_update_worker_nodes()` |
| 存储感知调度 | `sllm/schedulers/storage_aware_scheduler.py` | `StorageAwareScheduler(FcfsScheduler)`：`schedule()`、`get_migration_plans()`、`_get_model_loading_time()`、`_get_migration_target()` |
| 路由/自动伸缩 | `sllm/routers/roundrobin_router.py` | `RoundRobinRouter`：`auto_scaler()`、`inference()`、`fine_tuning()`、`_load_balancer_loop()`、`_auto_scaler_loop()`、`_create_instance()` / `_start_instance()` / `_stop_instance()` / `_finish_instance()` |
| 迁移路由 | `sllm/routers/migration_router.py` | `MigrationRouter(RoundRobinRouter)`：`inference()`、`execute_migration_plan()`、`get_instance_status()` |
| 推理实例 | `sllm/inference_instance.py` | `start_instance()`：按 `backend` 分发到 `VllmBackend` / `TransformersBackend` / `DummyBackend`，以 Ray detached actor 启动（`max_concurrency=10`）。微调实例对应 `sllm/fine_tuning_instance.py:start_ft_instance()` |
| vLLM 后端 | `sllm/backends/vllm_backend.py` | `VllmBackend`：`init_backend()`、`generate()`、`encode()`（embedding）、`get_current_tokens()` / `resume_kv_cache()`、`process_output()` |
| Transformers 后端 | `sllm/backends/transformers_backend.py` | `TransformersBackend`：`init_backend()`、`generate()`、`encode()`、`last_token_pool()`、`load_lora_adapter()`、`resume_generate()` |
| 微调后端 | `sllm/ft_backends/peft_lora_backend.py` | `PeftLoraBackend`：`fine_tuning()`、`_load_dataset()`；`FineTuningStatus` |
| CLI | `sllm/cli/clic.py` | Click 入口 `cli`：`deploy`（`--config/--backend/--target` 等）、`delete`、`start`（`--enable_storage_aware/--enable_migration`）、`status` |
| sllm-store Python | `sllm_store/sllm_store/` | `client.py:SllmStoreClient`（`load_into_cpu()` / `load_into_gpu()` / `unload_from_cpu()`，gRPC）；`server.py:StorageServicer`；`transformers.py`（`save_model` / `load_model` / `best_effort_load`）；`cli.py`（`sllm-store save/start/...`）；`proto/storage.proto`（`Storage` 服务：`LoadModelAsync / ConfirmModel / UnloadModel / ClearMem / RegisterModel / GetServerConfig`） |
| sllm-store C++ | `sllm_store/csrc/` | `checkpoint_store.cpp:CheckpointStore`（`RegisterModel` / `LoadModelFromDisk`，LRU 式 `TryFreeHost()`）；`model.cpp:Model`（`ToHost(num_threads)`，`open(..., O_DIRECT \| O_RDONLY)`，`cudaMemcpy`）；`pinned_memory_pool.cpp:PinnedMemoryPool`；`cuda_memory*.cpp`、`gpu_replica.cpp`；`checkpoint/checkpoint.cpp`（`tensor_writer.cpp`） |
| 部署 | `entrypoint.sh`、`Dockerfile`、`examples/docker/docker-compose.yml` | `entrypoint.sh` 按 `MODE=HEAD/WORKER` 分支：head 起 `ray start --head --port=6379 --resources='{"control_node": 1}'` 后 `exec sllm start`；worker 需 `WORKER_ID`，起 `sllm-store`（`--mem-pool-size 4GB --registration-required true`）。compose：`sllm_head`（`8343:8343` + `6379:6379`）+ `sllm_worker_0`（GPU 0，`${MODEL_FOLDER}:/models`）。另有 `examples/live_migration/`、`examples/storage_aware_scheduling/`、`examples/embedding/`、`examples/fine_tuning/` 各自的 compose/配置 |

## 实验性与边界部分（动手前必须知道）

- **AMD ROCm**：明确标注实验性。README：“AMD GPUs: ROCm 6.2+ (MI100, MI200 series) - Experimental”；
  `docs/store/rocm_quickstart.md` 称“no pre-built wheels for ROCm”，需从源码构建或用 `sllm_store/Dockerfile.rocm`
  （基于 `rocm/vllm-dev:base_ROCm-6.3.1_...`），且 ROCm < 6.2.0 有 GPU 显存泄漏（引 ROCm/HIP#3580）。
  `sllm_store/cmake/hipify.py` 说明 HIP 代码由 hipify 生成（`checkpoint_store_hip.cpp` 等未以源码形式入库）。
  CI 有 `rocm_test.yaml`（self-hosted amd，需 `ready-for-gpu-testing` 标签）。
- **其他硬件**：issue #312（Ascend 适配）维护者回复指向 Gitee 上的 openEuler 移植版 `gitee.com/openeuler/ServerlessLLM`，
  上游仓库本身无 Ascend 代码；issue #336（Ascend，2026-04-23）与 #312 主题重叠。
- **vLLM 集成是“打补丁”式**：必须应用 `sllm_store/vllm_patch/sllm_load.patch`（新增 `ServerlessLLMLoader`，`load_format=serverless_llm`），
  已测试版本为 vLLM 0.11.2（`requirements-worker.txt` 钉死 `vllm==0.11.2`；#320、#324）。vLLM 升级即需重新验证 patch。
- **SGLang**：仅存在于未合并的 open PR #278（`feat(backend): add SGLang backend`）和 #315（sglang weight loader，WIP），不在 `main` 代码里。
- **v1 重构**：`v1-beta` / `v1-batch` 分支上有 batch API（#337）、model downloader（#328）、RESTful API 调整（#325）等已合并工作，
  但均未合入 `main`。对 `main` 提 PR 时需确认目标分支仍是 `main`。

## 测试与贡献门槛

- 测试用 **pytest**（`requirements-test.txt`：`pytest>=8.3.2`、`pytest-asyncio>=0.23.8`）；
  `setup.py` 的 `extras["test"]` = worker 依赖 + pytest + pytest-asyncio。
- `tests/` 下：`backend_test/`（vllm / transformers / peft_lora 后端）、`new_cli_test/`（deploy/start/status/delete）、
  `inference_test/`（`store_test.py`、`inference_test.py`，按 `supported_models.json` 逐模型测）、`quantize_test/`、`replay_test/`（workload 回放）。
  `sllm_store/tests/python/`（`test_cli.py`、`test_save_model.py`、`test_save_vllm_model.py`、`test_save_lora.py`、`test_utils.py`）
  + `sllm_store/tests/cpp/`（`test_load_to_host.cpp`、`test_pinned_memory.cpp`，经 `ctest`）。
- **GPU 测试是门控的，不是默认跑的**：`backend.yaml`、`test_sllm_store.yaml`、`inference_test.yaml`、`quantize_test.yaml`、
  `rocm_test.yaml` 均运行在 self-hosted GPU runner（`[self-hosted, nvidia]` / `[self-hosted, amd]`），
  且 `if: contains(... labels, 'ready-for-gpu-testing')`——PR 必须被打上 `ready-for-gpu-testing` 标签才会跑 GPU 测试。
  CPU 上可直接跑的是 `cli_test.yaml`（`ubuntu-latest`，`pytest tests/new_cli_test`）和 `pre-commit-format.yml`。
  本地无 GPU 时，后端/推理/量化测试基本跑不了；`dummy` 后端（`sllm/backends/dummy_backend.py`）是唯一的无 GPU 调试路径（覆盖度未核实）。
- **无 DCO bot、无 DCO workflow**。`CONTRIBUTING.md` 只写“(Recommended) Sign off your commits: `git commit -s -m "feat: ..."`”——是建议，不是强制检查。
  提交格式要求 Angular 规范：`<type>: <summary>`，type 限 `build/ci/docs/feat/fix/perf/refactor/test/chore`；除 `docs` 外 body 必填且至少 20 字符。
- Format/lint：pre-commit（`.pre-commit-config.yaml`）= `pre-commit-hooks` + `codespell` + `clang-format`（C++，`.clang-format`）
  + `ruff` + `ruff-format`（`ruff==0.6.9`，`pyproject.toml` 中 `line-length=80`，lint 仅启用 isort 规则 `I`）。
  `requirements-lint.txt` 另含 `mypy==1.8.0`、`isort`，但 pre-commit 中未见 mypy hook。
- CI 门禁：PR 需过 `Linters`（markdown 链接检查）、`Formatting`（`pre-commit run --all-files`，含每日定时跑）以及按路径触发的对应测试 workflow；
  PR 模板要求勾选“已读 CONTRIBUTING / 已更新测试 / 已更新文档”。`CONTRIBUTING.md` 要求 Fork-and-Pull-Request、PR 标题遵循 Angular 格式、
  描述遵循 PR 模板并关联 issue；`/.github/ISSUE_TEMPLATE/` 仅有 `bug_report.yml`、`feature_request.yml`；`/.github/workflows/` 共 15 个 workflow。

## Issue 池分析

GitHub 的 `open_issues_count=45` 是含 PR 的口径，实际为 **33 个纯 issue + 12 个 open PR**。

- 12 个 open PR：#332（Prometheus 集成）、#315（sglang weight loader，WIP）、#303（server v1 refactor）、#301（同模型多 alias，WIP）、
  #295（shared memory 基础函数）、#291（quickstart 文档）、#287（example dtype）、#284（多 GPU 量化）、#278（SGLang 后端）、
  #225（transformers 后端流式生成）、#132（RAG demo 草稿）、#111（Colab quickstart）。最近更新为 #315（2026-01-26）。
- 33 个纯 issue 的标签统计（一个 issue 可多标签）：无标签 15；`Bug` 4（#331、#93、#78、#12）；
  `New feature` 6（#312、#176、#120、#114、#105、#83）；`Question` 6（#133、#117、#81、#75、#67、#14）；
  `Documentation` 1（#47）；`Good first issue` 1（#84）；`Priority 0` 2（#331、#114）、`Priority 1` 2（#176、#83）、`Priority 2` 2（#84、#12）。
  没有任何 open issue 带 `stale` 标签，尽管 `.github/workflows/stale.yml` 配置了 90 天标记 + 7 天关闭——stale bot 是否实际生效未核实。
- 时间分布：33 个纯 issue 中，2024 年的 15 个、2025 年的 14 个、2026 年的仅 4 个（#336、#331、#322、#321）。池子整体偏旧。

### 唯一 good first issue：#84（已指派、已停滞约 23 个月）

`Verify and add example for State Space Model (SSM) support in ServerlessLLM`，2024-09-13 由 `andrei3131` 创建，
标签 `Good first issue` + `Priority 2`，**已指派给 `febr8923`**，3 条评论，最后活动 2024-11-20。已独立核实：open、指派人仍为 `febr8923`。

- 正文任务：验证 SSM（如 Mamba）支持，分两步：(1) `sllm-store` 能正确转换与加载 SSM checkpoint；(2) `sllm` 能用 SSM 模型做推理。
  终态是给出一个端到端示例，类似 quickstart。
- 进展：`febr8923` 2024-11-06 认领；2024-11-20 汇报：`save_model()` / `load_model()` 对 Mamba、Jamba 可用，
  但 store quickstart 的 `model.generate(**inputs)` 会失败（`generate()` 不接受 `attention_mask`，只传 `input_ids` 可绕过）；
  `future-xy` 同日回复确认“`sllm-store` 已支持 SSM，问题在 transformers 后端的 `generate`”，要求提供复现代码并确认做到三步中的哪一步。
  此后再无任何活动约 23 个月，无关联 PR（按 “SSM” 搜索仅命中 #84 本身）。
- 是否仍然有效：形式上仍 open，但实质上已停滞且部分过时。三步任务中的第 1 步（store 保存/加载 SSM）按认领者汇报已验证可用；
  剩余是第 2 步（transformers 后端 `generate` 对 SSM）+ 第 3 步（端到端示例）。
  另注意 issue 正文引用的路径 `sllm/serve/backends/transformers_backend.py` 已不存在（2025-08-11 #296 扁平化重构后为 `sllm/backends/transformers_backend.py`）。
- **不能直接认领——已有指派人，且需先在 issue 下确认指派是否仍有效。**

### 备选 issue（只给事实，不断言“值得做”）

坦率结论：池中不存在同时满足“小 + 无人认领 + 近期有效”的 issue。以下是最接近的候选，各有明确短板：

1. **#336**（2026-04-23，最新、0 评论、无指派、无标签、正文为空）：`Can ServerlessLLM Store be modified and adapted for Ascend NPU cards?`——
   最新且无人认领，但不是小任务（硬件移植），且与 #312 主题重叠（#312 中维护者已指向 Gitee Ascend 移植版），做之前必须先确认是否重复。
2. **#120**（2024-10-25，2 评论、无指派、`New feature`）：`Support for Kubernetes (K8s) Deployment and Helm Chart`——
   无人认领、需求明确（仓库确实无 K8s 清单），但是大型功能（Helm chart + 部署），且已搁置约 2 年。
3. **#47**（2024-07-31，1 评论、`Documentation`）：`Docs: Improve onboarding for users by providing a diagram and an FAQ`——
   相对小（架构图 + FAQ），但已指派给 `future-xy`（维护者本人），且 2 年无活动；属于文档类，工程含金量有限。

其余无指派 issue 均不适合直接下手：#114 / #105（CPU-only 环境，大功能，2024 年）、#133 / #117 / #75（Question 类提问，非任务）、
#78（2024 年旧 Bug，报告者后续已按维护者建议解决，残留的“加载慢”追问源于当时 vLLM 后端未用自定义格式，现已过时，有效性存疑）。
另有多个近期的实质 Bug 已有维护者指派（如 #331、#322、#321 均指派 `future-xy`），不宜抢做。

## 维护者活跃度与风险（动手前最关键的一点）

- 合入 `main` 的最近 PR 是 #309（2026-03-06，release pipeline 修复）；`main` 最近 30 个提交覆盖 2025-06 至 2026-03。
  距核查日（2026-10-14）**`main` 已约 7 个月无合并**。
- 合入其他分支的最近活动：#328（model downloader）2026-05-04 合入 `v1-beta`；#337（batch API）2026-04-24 合入 `v1-batch`；
  #325 2025-12-29 合入 `v1-beta`。
- Issue 区维护者最近回复：#322（2025-12-16，`future-xy` 确认 Bug 可复现）、#321（2025-12-15，`future-xy` 回复加载带宽问题）、
  #312（2025-11-10，`future-xy` 回复 Ascend 移植指向 Gitee）。**2026 年以来未见维护者在 issue 区的新回复**（#336、#331 在 2026 年创建后 0 回复）。
- Open PR 最近更新为 #315（2026-01-26），其余多停留在 2025 年中。
- 结论：核心维护者 `future-xy`（Yao Fu）等人在 2025 年底前仍活跃回复，但 **2026 年 `main` 线基本停摆，开发转向未合入 `main` 的 `v1-*` 分支**。
  提 PR 前必须先确认维护者希望的落点分支（`main` 还是 `v1-beta`），否则有石沉大海风险。

## 动手前必须确认的两件事（未确认前不要开写）

1. **目标分支**：维护者希望新 PR 落在 `main` 还是 `v1-beta`？`main` 已 7 个月无合并，`v1-*` 才是实际开发线。
2. **#84 指派状态**：`febr8923` 的指派是否仍有效？若要接手，需先在 issue 下确认，而不是直接开写。

## 未核实清单

1. 未读 OSDI'24 论文全文；“论文主张”以 README 归纳为准，论文中的具体数字（如 300-700ms 加载 70B，issue #14 曾追问复现）未核实。
2. README 的 6-10 倍加速表、10 模型 / 1 GPU、数百 LoRA adapter 等数字均为项目方自测，未独立复现。
3. Stale bot（`stale.yml`）是否实际启用未核实（池中无 `stale` 标签与之矛盾）。
4. `dummy` 后端的实际覆盖度、CPU-only demo 是否可用未实测。
5. 未核查 Discord / 微信群等站外讨论；`future-xy` 等人的 2026 年动向仅以 GitHub 可见活动为准。
6. 分支列表通过公开 API 获取（connector 只读集无 `list_branches`）；若 fork 有未推送的本地分支，不可见。

## 一句话总结

仓库架构清晰、核心（快加载 + 存储感知调度 + LoRA 微调统一）真实落地，但 **issue 池陈旧、唯一 good first issue（#84）已指派且停滞约 23 个月、
且 `main` 线自 2026-03 起无合并、开发转向未合入 `main` 的 `v1-*` 分支**——动手前最关键的一步是向维护者确认目标分支与 #84 指派状态，而不是直接开写。
