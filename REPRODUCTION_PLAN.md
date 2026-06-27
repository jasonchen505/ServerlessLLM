# ServerlessLLM 8卡4090完整复现计划

> 硬件环境：8x NVIDIA RTX 4090 (24GB GDDR6X each)
> 目标：完整复现ServerlessLLM的所有核心功能

---

## 目录

- [第一章：硬件评估与可行性分析](#第一章硬件评估与可行性分析)
- [第二章：环境搭建](#第二章环境搭建)
- [第三章：分阶段复现计划](#第三章分阶段复现计划)
- [第四章：学习路径设计](#第四章学习路径设计)
- [第五章：预期成果与验收标准](#第五章预期成果与验收标准)

---

## 第一章：硬件评估与可行性分析

### 1.1 4090硬件规格

| 参数 | 规格 | 对ServerlessLLM的影响 |
|------|------|----------------------|
| **显存** | 24GB GDDR6X | 可运行7B-13B模型（FP16） |
| **CUDA Compute Capability** | 8.9 | 完全兼容（要求7.0+） |
| **显存带宽** | 1008 GB/s | 高带宽，利于推理 |
| **PCIe** | PCIe 4.0 x16 | 32 GB/s，影响CPU-GPU传输 |
| **NVLink** | 需要桥接器 | 无NVLink时GPU间通信通过PCIe |
| **TDP** | 450W | 8卡总功耗3600W，需要强大电源 |

### 1.2 可运行模型规模

```
4090 24GB显存可运行的模型（FP16）：
├── 7B模型：~14GB显存 ✅ 轻松运行
├── 13B模型：~26GB显存 ⚠️ 需要量化或device_map
├── 32B模型：~64GB显存 ❌ 需要多卡张量并行
└── 70B模型：~140GB显存 ❌ 需要4-8卡张量并行

推荐复现模型：
├── 入门：Qwen2.5-1.5B, OPT-1.3B（单卡）
├── 进阶：Qwen2.5-7B, Llama-3-8B（单卡）
├── 挑战：Qwen2.5-14B（双卡张量并行）
└── 极限：Qwen2.5-32B（4卡张量并行）
```

### 1.3 复现可行性评估

| 功能 | 可行性 | 说明 |
|------|--------|------|
| **基础模型加载** | ✅ 完全可行 | 4090支持CUDA 7.0+ |
| **多模型GPU复用** | ✅ 完全可行 | 核心功能，24GB足够 |
| **Storage-Aware调度** | ✅ 完全可行 | 与GPU型号无关 |
| **Live Migration** | ✅ 完全可行 | 需要至少2个Worker |
| **LoRA微调** | ✅ 完全可行 | 4090支持 |
| **vLLM后端** | ✅ 完全可行 | vLLM支持4090 |
| **量化模型** | ✅ 完全可行 | INT8/INT4节省显存 |
| **分布式调度** | ✅ 完全可行 | 8卡可模拟8个Worker |

### 1.4 硬件配置建议

#### 方案A：8个独立Worker（推荐）
```
用途：模拟多机多卡场景
配置：
├── Head Node：CPU + 内存（不需要GPU）
├── Worker 0-7：每个Worker绑定1个GPU
└── 适用场景：Live Migration、多模型调度测试

优势：
├── 最大化模拟分布式场景
├── 每个Worker独立，故障隔离
└── 可以测试8个模型同时运行

劣势：
├── 无法运行大模型（>24GB）
└── GPU间通信通过PCIe，较慢
```

#### 方案B：2个Worker，每个4卡
```
用途：运行大模型（32B-70B）
配置：
├── Worker 0：GPU 0-3（张量并行）
├── Worker 1：GPU 4-7（张量并行）
└── 适用场景：大模型推理、性能测试

优势：
├── 可运行32B-70B模型
├── GPU间NVLink/PCIe通信更快
└── 更接近生产环境

劣势：
├── 无法测试多Worker调度
└── 模型数量受限
```

#### 方案C：混合配置（最灵活）
```
用途：全面测试所有功能
配置：
├── Worker 0-3：单卡Worker（4个独立Worker）
├── Worker 4-5：双卡Worker（张量并行）
├── Worker 6-7：4卡Worker（大模型）
└── 根据测试需求动态调整

优势：
├── 灵活性最高
├── 可以测试各种场景
└── 资源利用率高

劣势：
├── 配置复杂
└── 需要手动管理资源
```

---

## 第二章：环境搭建

### 2.1 基础环境准备

#### 系统要求
```bash
# 操作系统
Ubuntu 22.04 LTS 或 CentOS 8+

# CUDA版本
CUDA 12.1+（推荐12.1.1）

# Docker
Docker 24.0+
Docker Compose v2.20+

# NVIDIA Container Toolkit
nvidia-container-toolkit 1.14+
```

#### 环境检查脚本
```bash
#!/bin/bash
# check_environment.sh

echo "=== 系统信息 ==="
uname -a
cat /etc/os-release | grep PRETTY_NAME

echo -e "\n=== GPU信息 ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo -e "\n=== CUDA版本 ==="
nvcc --version

echo -e "\n=== Docker版本 ==="
docker --version
docker compose version

echo -e "\n=== NVIDIA Container Toolkit ==="
nvidia-container-cli --version

echo -e "\n=== 磁盘空间 ==="
df -h / | tail -1

echo -e "\n=== 内存信息 ==="
free -h | head -2

echo -e "\n=== 网络检查 ==="
ping -c 1 huggingface.co > /dev/null 2>&1 && echo "HuggingFace: OK" || echo "HuggingFace: FAILED"
```

### 2.2 项目克隆与构建

```bash
# 1. 克隆项目
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM

# 2. 创建模型存储目录
mkdir -p ~/models
export MODEL_FOLDER=~/models

# 3. 构建Docker镜像（首次需要较长时间）
docker compose -f examples/docker/docker-compose.yml build

# 4. 验证构建
docker images | grep serverlessllm
```

### 2.3 环境变量配置

```bash
# ~/.bashrc 或 ~/.zshrc
export SERVERLESSLLM_HOME=/path/to/ServerlessLLM
export MODEL_FOLDER=~/models
export STORAGE_PATH=~/models
export HF_TOKEN=your_huggingface_token  # 如果需要下载gated模型

# CUDA相关
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST="8.9"  # 4090的compute capability
```

---

## 第三章：分阶段复现计划

### 阶段0：环境验证（Day 1）

#### 目标
验证硬件环境和基础依赖是否正常工作

#### 任务清单
```bash
# 1. 运行环境检查
bash check_environment.sh

# 2. 测试CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 3. 测试Docker GPU支持
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# 4. 测试HuggingFace访问
python -c "from huggingface_hub import snapshot_download; print('HF OK')"
```

#### 验收标准
- [x] 8个GPU都被识别
- [x] CUDA版本 >= 12.1
- [x] Docker可以访问GPU
- [x] 可以访问HuggingFace

---

### 阶段1：单机基础功能（Day 2-3）

#### 目标
复现ServerlessLLM的核心功能：模型加载和推理

#### 任务清单

**Day 2：单Worker部署**
```bash
# 1. 启动单Worker集群
cd examples/docker
docker compose up -d

# 2. 等待集群就绪
docker logs -f sllm_head  # 看到"Cluster initialized"即可

# 3. 下载并转换小模型（快速验证）
docker exec sllm_worker_0 sllm-store save \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --backend transformers

# 4. 部署模型
docker exec sllm_head sllm deploy \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --backend transformers

# 5. 测试推理
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Day 3：多模型部署**
```bash
# 1. 下载多个模型
for model in Qwen/Qwen2.5-0.5B-Instruct facebook/opt-1.3b; do
    docker exec sllm_worker_0 sllm-store save \
        --model $model --backend transformers
done

# 2. 部署多个模型
for model in Qwen/Qwen2.5-0.5B-Instruct facebook/opt-1.3b; do
    docker exec sllm_head sllm deploy \
        --model $model --backend transformers
done

# 3. 测试多模型推理
for model in Qwen/Qwen2.5-0.5B-Instruct facebook/opt-1.3b; do
    curl -s http://localhost:8343/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}" \
      | jq '.model, .choices[0].message.content'
done

# 4. 查看模型状态
docker exec sllm_head sllm status
```

#### 学习重点
1. **模型转换流程**：理解sllm-store save做了什么
2. **部署流程**：理解sllm deploy的完整流程
3. **请求路由**：理解请求如何到达正确的模型

#### 验收标准
- [x] 成功部署2个以上模型
- [x] 所有模型都能正常推理
- [x] 理解模型转换和部署流程

---

### 阶段2：性能测试与对比（Day 4-5）

#### 目标
复现论文中的6-10x加速，并理解性能瓶颈

#### 任务清单

**Day 4：加载性能测试**
```bash
# 1. 运行benchmark
cd benchmarks
python test_loading.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-format sllm \
    --model-dir ~/models \
    --num-replicas 5 \
    --benchmark-type random

python test_loading.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-format safetensors \
    --model-dir ~/models \
    --num-replicas 5 \
    --benchmark-type random

# 2. 对比结果
cat results/*.json | jq '.[] | {model_name, loading_time}'
```

**Day 5：多模型并发测试**
```bash
# 1. 编写并发测试脚本
cat > test_concurrent.py << 'EOF'
import asyncio
import aiohttp
import time

async def send_request(session, model, prompt):
    async with session.post(
        "http://localhost:8343/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
    ) as resp:
        return await resp.json()

async def main():
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "facebook/opt-1.3b",
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(10):
            model = models[i % len(models)]
            tasks.append(send_request(session, model, f"Hello {i}"))
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        print(f"10 requests completed in {elapsed:.2f}s")
        print(f"Average latency: {elapsed/10:.2f}s")

asyncio.run(main())
EOF

# 2. 运行并发测试
python test_concurrent.py
```

#### 学习重点
1. **性能瓶颈分析**：理解IO-bound vs Compute-bound
2. **缓存机制**：理解Pinned Memory Pool的作用
3. **并发处理**：理解Router的负载均衡

#### 验收标准
- [x] 复现加载性能对比（SLLM vs SafeTensors）
- [x] 理解性能瓶颈在哪里
- [x] 理解缓存机制如何工作

---

### 阶段3：Storage-Aware调度（Day 6-7）

#### 目标
复现Storage-Aware调度，并理解调度决策

#### 任务清单

**Day 6：配置Storage-Aware调度**
```bash
# 1. 修改docker-compose启用storage-aware
# examples/storage_aware_scheduling/docker-compose.yml
services:
  sllm_head:
    command: ["--enable-storage-aware"]
    # ...

# 2. 启动集群
cd examples/storage_aware_scheduling
docker compose up -d

# 3. 部署模型到不同节点
# config-opt-2.7b.json 部署到 node 0
# config-opt-1.3b.json 部署到 node 1
docker exec sllm_head sllm deploy --config config-opt-2.7b.json
docker exec sllm_head sllm deploy --config config-opt-1.3b.json
```

**Day 7：观察调度决策**
```bash
# 1. 查看调度日志
docker logs sllm_head 2>&1 | grep -E "(schedule|allocated|loading)"

# 2. 理解调度决策
# 当请求 opt-2.7b 时，调度器会选择哪个节点？
# 当请求 opt-1.3b 时，调度器会选择哪个节点？

# 3. 测试数据本地性
# 先请求 opt-2.7b，观察它被加载到哪个节点
# 再请求 opt-1.3b，观察调度决策
```

#### 学习重点
1. **调度算法**：理解FCFS和Storage-Aware的区别
2. **数据本地性**：理解为什么模型位置很重要
3. **加载时间估算**：理解`_get_model_loading_time`的计算

#### 验收标准
- [x] 成功配置Storage-Aware调度
- [x] 理解调度决策日志
- [x] 能够解释为什么模型被加载到特定节点

---

### 阶段4：Live Migration（Day 8-9）

#### 目标
复现Live Migration功能，并理解迁移机制

#### 任务清单

**Day 8：配置Live Migration**
```bash
# 1. 使用live_migration示例
cd examples/live_migration

# 2. 启动带migration的集群
docker compose -f docker-compose.yml -f enable-migration.yml up -d

# 3. 部署模型
docker exec sllm_head sllm deploy --config config-qwen-1.5b.json
docker exec sllm_head sllm deploy --config config-qwen-3b.json
```

**Day 9：触发和观察迁移**
```bash
# 1. 触发迁移场景
# 并行发送两个请求，第一个是大模型，第二个是小模型
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a long story"}],
    "max_tokens": 1024
  }' &

sleep 3

curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'

# 2. 观察迁移日志
docker logs sllm_head 2>&1 | grep -E "(migration|migrate|Migration)"
```

#### 学习重点
1. **迁移触发条件**：理解什么时候会触发迁移
2. **KV Cache迁移**：理解如何迁移推理状态
3. **迁移开销**：理解迁移的时间成本

#### 验收标准
- [x] 成功触发Live Migration
- [x] 理解迁移日志
- [x] 能够解释迁移的收益和开销

---

### 阶段5：LoRA微调（Day 10-11）

#### 目标
复现LoRA微调功能，并理解统一推理+微调架构

#### 任务清单

**Day 10：LoRA推理**
```bash
# 1. 下载LoRA适配器（如果有现成的）
# 或者使用示例LoRA

# 2. 部署带LoRA的模型
docker exec sllm_head sllm deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend transformers \
    --enable-lora \
    --lora-adapters "adapter1=path/to/adapter1 adapter2=path/to/adapter2"

# 3. 使用LoRA推理
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "lora_adapter_name": "adapter1"
  }'
```

**Day 11：LoRA微调**
```bash
# 1. 提交微调任务
curl -X POST http://localhost:8343/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "ft_backend": "peft_lora",
    "backend_config": {
      "training_data": "path/to/data.jsonl",
      "num_epochs": 3,
      "learning_rate": 2e-4
    }
  }'

# 2. 查看微调状态
curl http://localhost:8343/v1/fine_tuning/jobs/{job_id}
```

#### 学习重点
1. **LoRA原理**：理解低秩适配的数学原理
2. **动态加载**：理解如何在推理时切换LoRA
3. **资源管理**：理解微调和推理如何共享资源

#### 验收标准
- [x] 成功使用LoRA推理
- [x] 理解LoRA的加载机制
- [x] 能够解释LoRA的优势和局限

---

### 阶段6：vLLM后端（Day 12-13）

#### 目标
复现vLLM后端，并理解vLLM的优化

#### 任务清单

**Day 12：vLLM后端部署**
```bash
# 1. 转换模型为vLLM格式
docker exec sllm_worker_0 sllm-store save \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm

# 2. 部署vLLM后端模型
docker exec sllm_head sllm deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --num-gpus 1

# 3. 测试vLLM推理
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Day 13：vLLM vs Transformers对比**
```bash
# 1. 性能对比脚本
cat > compare_backends.py << 'EOF'
import time
import requests

def benchmark(model, backend, num_requests=10):
    url = "http://localhost:8343/v1/chat/completions"
    latencies = []
    
    for i in range(num_requests):
        start = time.time()
        resp = requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": f"Hello {i}"}]
        })
        latencies.append(time.time() - start)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"{backend}: {avg_latency:.3f}s average")

# 2. 运行对比
benchmark("Qwen/Qwen2.5-7B-Instruct_transformers", "Transformers")
benchmark("Qwen/Qwen2.5-7B-Instruct_vllm", "vLLM")
EOF

python compare_backends.py
```

#### 学习重点
1. **vLLM架构**：理解PagedAttention和Continuous Batching
2. **性能差异**：理解vLLM在高并发下的优势
3. **后端选择**：理解什么时候用vLLM，什么时候用Transformers

#### 验收标准
- [x] 成功部署vLLM后端
- [x] 理解vLLM的核心优化
- [x] 能够对比两个后端的优劣

---

### 阶段7：源码分析与修改（Day 14-17）

#### 目标
深入理解源码，并进行有意义的修改

#### 任务清单

**Day 14-15：阅读核心代码**
```python
# 1. Controller代码分析
# sllm/controller.py
# 重点理解：
# - 模型注册流程
# - 微调任务调度
# - 状态管理

# 2. Router代码分析
# sllm/routers/roundrobin_router.py
# 重点理解：
# - 负载均衡算法
# - 自动扩缩容
# - 实例生命周期

# 3. Scheduler代码分析
# sllm/schedulers/storage_aware_scheduler.py
# 重点理解：
# - 调度算法
# - 迁移决策
# - 资源估算
```

**Day 16-17：代码修改练习**
```python
# 练习1：添加自定义调度策略
# 在 sllm/schedulers/ 目录下创建 new_scheduler.py

class LeastLoadedScheduler(FcfsScheduler):
    """选择负载最低的节点"""
    async def schedule(self, model_name, num_gpus, worker_nodes, ...):
        options = []
        for node_id, node_info in worker_nodes.items():
            # 计算节点负载
            load = node_info["total_gpu"] - node_info["free_gpu"]
            latency = self._estimate_latency(model_name, node_id)
            options.append(AllocationPlan(node_id, latency, load))
        
        # 选择负载最低的节点
        return min(options, key=lambda x: (x.load, x.latency))

# 练习2：添加监控指标
# 在 Router 中添加 Prometheus 指标

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'sllm_requests_total',
    'Total requests',
    ['model', 'status']
)

REQUEST_LATENCY = Histogram(
    'sllm_request_latency_seconds',
    'Request latency',
    ['model']
)
```

#### 学习重点
1. **代码架构**：理解Ray Actor模型
2. **异步编程**：理解asyncio在系统中的应用
3. **扩展点**：理解如何扩展系统功能

#### 验收标准
- [x] 能够画出系统架构图
- [x] 理解核心组件的交互
- [x] 完成至少一个代码修改练习

---

### 阶段8：压力测试与优化（Day 18-20）

#### 目标
测试系统极限，并进行性能优化

#### 任务清单

**Day 18-19：压力测试**
```bash
# 1. 多模型压力测试
cat > stress_test.py << 'EOF'
import asyncio
import aiohttp
import time
import random

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "facebook/opt-1.3b",
]

async def send_request(session, model):
    async with session.post(
        "http://localhost:8343/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50
        }
    ) as resp:
        return await resp.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            model = random.choice(MODELS)
            tasks.append(send_request(session, model))
        
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
        
        success = sum(1 for r in results if not isinstance(r, Exception))
        print(f"100 requests: {success} success, {100-success} failed")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {success/elapsed:.2f} req/s")

asyncio.run(main())
EOF

# 2. 运行压力测试
python stress_test.py
```

**Day 20：性能优化**
```python
# 1. 分析性能瓶颈
# 使用cProfile分析Python代码
python -m cProfile -o profile.stats your_script.py

# 2. 分析GPU使用
nvidia-smi dmon -s pucvmet -d 1

# 3. 分析IO使用
iostat -x 1
```

#### 学习重点
1. **性能瓶颈**：识别CPU/GPU/IO瓶颈
2. **优化策略**：理解如何优化不同类型的瓶颈
3. **系统限制**：理解系统的理论极限

#### 验收标准
- [x] 完成100+请求的压力测试
- [x] 识别出性能瓶颈
- [x] 提出并实施至少一个优化

---

## 第四章：学习路径设计

### 4.1 学习路线图

```
Week 1: 基础功能复现
├── Day 1: 环境搭建
├── Day 2-3: 单机基础功能
├── Day 4-5: 性能测试
└── Day 6-7: Storage-Aware调度

Week 2: 高级功能与源码
├── Day 8-9: Live Migration
├── Day 10-11: LoRA微调
├── Day 12-13: vLLM后端
└── Day 14-17: 源码分析与修改

Week 3: 优化与总结
├── Day 18-20: 压力测试与优化
└── Day 21: 总结与文档
```

### 4.2 每日学习流程

```
上午（3-4小时）：
├── 1. 复现前一天的内容（30分钟）
├── 2. 学习新内容（2小时）
├── 3. 动手实践（1-1.5小时）
└── 4. 记录问题和收获

下午（3-4小时）：
├── 1. 继续实践（1.5小时）
├── 2. 阅读源码（1小时）
├── 3. 总结笔记（30分钟）
└── 4. 规划明天（30分钟）
```

### 4.3 学习资源

| 资源 | 用途 | 优先级 |
|------|------|--------|
| **项目README** | 快速上手 | ⭐⭐⭐⭐⭐ |
| **OSDI'24论文** | 理解设计思想 | ⭐⭐⭐⭐⭐ |
| **架构博客** | 理解系统架构 | ⭐⭐⭐⭐ |
| **示例代码** | 学习使用方式 | ⭐⭐⭐⭐ |
| **源码注释** | 理解实现细节 | ⭐⭐⭐ |
| **Discord社区** | 问题解答 | ⭐⭐⭐ |

---

## 第五章：预期成果与验收标准

### 5.1 功能验收清单

| 功能 | 验收标准 | 优先级 |
|------|----------|--------|
| **模型加载** | 成功加载3个以上模型 | P0 |
| **模型推理** | 所有模型正常推理 | P0 |
| **GPU复用** | 1个GPU服务3个模型 | P0 |
| **自动扩缩容** | 模型实例数随请求变化 | P1 |
| **Storage-Aware** | 理解调度决策 | P1 |
| **Live Migration** | 成功触发迁移 | P1 |
| **LoRA** | 成功使用LoRA推理 | P2 |
| **vLLM** | 成功部署vLLM后端 | P2 |
| **性能测试** | 复现加载性能对比 | P1 |
| **源码修改** | 完成一个修改练习 | P2 |

### 5.2 学习成果验收

| 成果 | 验收标准 | 优先级 |
|------|----------|--------|
| **系统架构** | 能够画出完整架构图 | P0 |
| **核心组件** | 能够解释每个组件的作用 | P0 |
| **调度算法** | 能够解释FCFS和Storage-Aware | P1 |
| **性能优化** | 能够解释6-10x加速原理 | P1 |
| **工程实践** | 能够部署和运维系统 | P1 |
| **源码理解** | 能够解释核心代码逻辑 | P2 |

### 5.3 面试准备验收

| 能力 | 验收标准 | 优先级 |
|------|----------|--------|
| **项目介绍** | 1分钟/5分钟版本 | P0 |
| **技术深度** | 能够回答底层原理问题 | P0 |
| **实验能力** | 能够设计验证实验 | P1 |
| **问题定位** | 能够排查常见问题 | P1 |
| **工程落地** | 能够讨论部署方案 | P1 |
| **业务理解** | 能够讨论适用场景 | P2 |

---

## 附录：常用命令速查

### Docker命令
```bash
# 启动集群
docker compose up -d

# 查看日志
docker logs -f sllm_head
docker logs -f sllm_worker_0

# 进入容器
docker exec -it sllm_head bash
docker exec -it sllm_worker_0 bash

# 停止集群
docker compose down
```

### ServerlessLLM命令
```bash
# 模型管理
sllm deploy --model MODEL_NAME --backend transformers
sllm delete MODEL_NAME
sllm status

# 模型转换
sllm-store save --model MODEL_NAME --backend transformers
sllm-store save --model MODEL_NAME --backend vllm

# 启动服务
sllm start --enable-storage-aware --enable-migration
```

### 调试命令
```bash
# GPU状态
nvidia-smi
watch -n 1 nvidia-smi

# 系统状态
top
htop
iostat -x 1

# Ray状态
ray status
ray list actors
```

---

## 总结

这个复现计划分为8个阶段，预计需要3周时间完成。每个阶段都有明确的目标、任务清单和验收标准。通过这个计划，你将：

1. **完整复现**ServerlessLLM的所有核心功能
2. **深入理解**系统的设计原理和实现细节
3. **掌握技能**LLM推理系统的部署和优化
4. **准备面试**能够回答技术面试的各类问题

建议按照计划逐步推进，不要跳过任何阶段。每个阶段结束后，对照验收标准检查自己的进度。

祝复现顺利！
