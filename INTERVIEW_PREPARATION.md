# ServerlessLLM 面试准备文档

> 针对 LLM & Agent 应用 / 后训练 & Infra 方向的面试准备
> 适用于：LLM算法实习（主算法辅Infra）

---

## 目录

- [第一部分：项目概述与核心价值](#第一部分项目概述与核心价值)
- [第二部分：系统架构深度解析](#第二部分系统架构深度解析)
- [第三部分：核心技术原理](#第三部分核心技术原理)
- [第四部分：面试高频考察点](#第四部分面试高频考察点)
- [第五部分：算法岗相关的考察点](#第五部分算法岗相关的考察点)
- [第六部分：模拟面试Q&A](#第六部分模拟面试qa)
- [第七部分：如何介绍这个项目](#第七部分如何介绍这个项目)

---

## 第一部分：项目概述与核心价值

### 1.1 项目定位

**一句话概括**：ServerlessLLM 是一个支持多模型共享GPU的高性能LLM推理系统，核心目标是**用1块GPU服务10个模型**，同时保持低延迟。

**核心数据**：
- 模型加载速度比 SafeTensors 快 **6-10x**
- 支持在单GPU上运行 **10+** 个模型
- 发表在 **OSDI'24**（操作系统顶会）

### 1.2 解决的核心问题

| 问题 | 传统方案 | ServerlessLLM |
|------|----------|---------------|
| 多模型部署 | 每个模型独占GPU，成本高 | GPU复用，按需加载 |
| 冷启动延迟 | 从磁盘加载模型慢（20s+） | 优化存储格式，3s内加载 |
| 资源利用率 | GPU空闲时仍占用 | 自动缩容到0 |
| 模型切换 | 卸载再加载，延迟高 | 快速切换，智能调度 |

### 1.3 三大核心创新

1. **超快Checkpoint加载**：自定义存储格式 + O_DIRECT I/O + pinned memory
2. **GPU复用**：多模型共享GPU，快速切换 + 智能调度
3. **统一推理+微调**：LoRA微调与推理无缝集成

---

## 第二部分：系统架构深度解析

### 2.1 整体架构（三层）

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌─────────┐  ┌─────────────────────────────────────┐  │
│  │   CLI   │  │         API Gateway (FastAPI)        │  │
│  └─────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Control Plane                         │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │  Controller   │  │   Scheduler   │  │Store Manager│  │
│  │  (Ray Actor)  │  │  (Ray Actor)  │  │ (Ray Actor) │  │
│  └──────────────┘  └───────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Data Plane                            │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │    Router     │  │   Backend     │  │  Sllm Store │  │
│  │  (Ray Actor)  │  │  (Ray Actor)  │  │   (C++)     │  │
│  └──────────────┘  └───────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件详解

#### Controller (`sllm/controller.py`)
**职责**：系统的"大脑"，管理所有模型的生命周期

```python
class SllmController:
    async def register(self, model_config):  # 注册模型
    async def delete(self, model_name):       # 删除模型
    async def update(self, model_name, config): # 更新配置
    async def register_ft_job(self, job_config): # 注册微调任务
```

**关键点**：
- 使用Ray Actor实现分布式
- 通过asyncio实现高并发
- 管理模型元数据和路由器

#### Router (`sllm/routers/roundrobin_router.py`)
**职责**：请求路由和负载均衡

```python
class RoundRobinRouter(SllmRouter):
    async def inference(self, request_data, action):  # 推理请求
    async def fine_tuning(self, request_data):         # 微调请求
    async def _auto_scaler_loop(self):                 # 自动扩缩容
    async def _load_balancer_loop(self):               # 负载均衡
```

**核心机制**：
- Round-Robin负载均衡
- 自动扩缩容（基于请求量）
- 实例池管理（starting/ready/deleting状态机）

#### Scheduler (`sllm/schedulers/storage_aware_scheduler.py`)
**职责**：决定模型加载到哪个节点

```python
class StorageAwareScheduler(FcfsScheduler):
    async def schedule(self, model_name, num_gpus, ...):
    async def get_migration_plans(self, ...):
    def _get_model_loading_time(self, ...):
```

**调度策略**：
1. **FCFS（先来先服务）**：简单公平
2. **Storage-Aware**：考虑数据本地性
   - 模型已在内存中？→ PCIe带宽加载
   - 模型在磁盘上？→ 磁盘带宽加载
   - 需要迁移？→ 计算迁移成本

#### Store Manager (`sllm/store_manager.py`)
**职责**：管理模型存储和加载

```python
class StoreManager:
    async def register(self, model_config):      # 注册模型
    async def load_to_host(self, node_id, model): # 加载到内存
    async def get_store_info(self):                # 获取存储信息
```

**存储层次**：
```
GPU显存 ← PCIe ← 主内存（Pinned） ← Disk（NVMe SSD）
```

### 2.3 请求处理流程

```
1. 用户请求 → API Gateway
2. API Gateway → Router（根据模型名路由）
3. Router → 选择可用Backend Instance
4. Backend → 处理推理请求
5. Backend → 返回结果给用户
```

**冷启动流程**：
```
1. Router发现无可用实例
2. Router → Scheduler请求资源
3. Scheduler → 选择最优节点（考虑数据本地性）
4. StoreManager → 从磁盘加载模型到内存
5. Backend → 从内存加载到GPU
6. Backend → 处理请求
```

---

## 第三部分：核心技术原理

### 3.1 超快模型加载（6-10x加速）

#### 问题：为什么传统加载慢？

```python
# 传统方式（SafeTensors）
model = AutoModelForCausalLM.from_pretrained("model_path")
# 1. 从磁盘读取文件（随机IO）
# 2. 反序列化为Python对象
# 3. 拷贝到GPU
```

**瓶颈**：
- 随机IO读取
- Python对象开销
- 多次内存拷贝

#### ServerlessLLM的解决方案

**1. 自定义存储格式**
```python
# sllm_store/torch.py
def save_dict(state_dict, model_path):
    # 1. 将所有tensor保存为连续的二进制文件
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)
    
    # 2. 保存tensor索引（offset, size, shape, stride, dtype）
    tensor_index = {}
    for name, param in state_dict.items():
        tensor_index[name] = (offset, size, shape, stride, dtype)
    
    # 3. 保存索引文件
    json.dump(tensor_index, open("tensor_index.json", "w"))
```

**2. O_DIRECT I/O**
```c++
// 绕过OS Page Cache，直接读取磁盘
int fd = open(path, O_RDONLY | O_DIRECT);
read(fd, buffer, size);  // 直接DMA到用户空间
```

**3. Pinned Memory + DMA**
```python
# 预分配pinned memory pool
cuda_memory_ptrs = allocate_cuda_memory(device_memory)

# 直接从pinned memory DMA到GPU
# 无需经过Page Cache
```

**4. 并行加载**
```python
# 多线程并行读取不同tensor
# 利用NVMe SSD的并行能力
```

#### 性能对比

| 模型 | SafeTensors | ServerlessLLM | 加速比 |
|------|-------------|---------------|--------|
| Qwen3-32B | 20.6s | 3.2s | 6.4x |
| DeepSeek-R1-32B | 19.1s | 3.2s | 5.9x |
| Llama-3.1-8B | 4.4s | 0.7s | 6.5x |

### 3.2 GPU复用与调度

#### 核心思想：Serverless的GPU共享

```
传统方案：
Model A → GPU 0 (独占)
Model B → GPU 1 (独占)
Model C → GPU 2 (独占)

ServerlessLLM：
Model A, B, C → GPU 0 (按需加载)
```

#### 调度算法

**FCFS（先来先服务）**：
```python
async def _control_loop(self):
    while self.running:
        # 获取所有等待的请求
        loading_requests = self._get_loading_requests()
        
        # 按到达顺序处理
        for request in loading_requests:
            node_id = self._allocate_resource(request)
            await self._load_model(node_id, request.model_name)
```

**Storage-Aware调度**：
```python
async def schedule(self, model_name, num_gpus, ...):
    options = []
    for node_id, node_info in worker_nodes.items():
        # 计算加载时间
        if model_name in pinned_memory_pool:
            # 模型已在内存中，用PCIe带宽
            latency = model_size / pcie_bandwidth
        else:
            # 模型在磁盘上，用磁盘带宽
            latency = waiting_time + model_size / disk_bandwidth
        
        options.append(AllocationPlan(node_id, latency))
    
    # 选择延迟最低的节点
    return min(options, key=lambda x: x.latency)
```

#### 自动扩缩容

```python
async def auto_scaler(auto_scaling_metrics, auto_scaling_config):
    request_count = auto_scaling_metrics["request_count"]
    min_instances = auto_scaling_config["min_instances"]
    max_instances = auto_scaling_config["max_instances"]
    target_ongoing_requests = auto_scaling_config["target"]
    
    # 基于请求量计算期望实例数
    desired_instances = (request_count + target_ongoing_requests - 1) // target_ongoing_requests
    
    # 限制在min/max范围内
    return min(max_instances, max(min_instances, desired_instances))
```

### 3.3 Live Migration（实时迁移）

#### 场景：资源争用时的模型迁移

```
场景：
- Node 0: 运行 Model A（大模型，正在推理）
- Node 0: 请求 Model B（小模型，需要低延迟）
- Node 1: 空闲

问题：Model B需要等待Model A完成吗？

解决方案：将Model A迁移到Node 1
```

#### 迁移流程

```python
async def get_migration_plans(self, ...):
    # 1. 找到可迁移的实例
    migratable_instances = self._find_migratable_instances()
    
    # 2. 使用DP算法选择最优迁移方案
    dp = [[MigrationPlans() for _ in range(gpu_shortage + 1)] 
          for _ in range(numInstances + 1)]
    
    # 3. 考虑迁移成本（KV Cache传输时间）
    for i in range(1, numInstances + 1):
        for j in range(gpu_shortage + 1):
            # 计算迁移延迟
            migration_latency = resuming_latency + loading_time
            
            # 更新DP表
            if total_latency < dp[i-1][j].total_latency:
                dp[i][j] = dp[i-1][j-n_gpus] + migration_plan
    
    return optimal_plans
```

#### KV Cache迁移

```python
async def execute_migration_plan(self, migration_plan):
    # 1. 获取当前推理状态
    current_tokens = await instance.get_current_tokens()
    
    # 2. 在目标节点创建新实例
    new_instance = await self._create_instance(target_node)
    
    # 3. 迁移KV Cache
    await new_instance.resume_kv_cache(current_tokens)
    
    # 4. 切换请求到新实例
    await self._switch_requests(old_instance, new_instance)
    
    # 5. 销毁旧实例
    await self._shutdown_instance(old_instance)
```

### 3.4 LoRA微调集成

#### 统一推理+微调架构

```python
class TransformersBackend:
    def load_lora_adapter(self, lora_name, lora_path):
        # 动态加载LoRA适配器
        self.model = load_lora(
            self.model, lora_name, lora_path,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
    
    def generate(self, request_data):
        # 推理时可以指定使用哪个LoRA
        lora_adapter_name = request_data.get("lora_adapter_name")
        if lora_adapter_name:
            generate_kwargs["adapter_names"] = [lora_adapter_name]
        
        outputs = self.model.generate(**inputs, **generate_kwargs)
```

#### Serverless LoRA微调

```python
async def fine_tuning(self, request_data):
    # 1. 创建微调专用实例
    instance_id = await self._create_ft_instance()
    
    # 2. 等待实例就绪
    await self._wait_for_instance(instance_id)
    
    # 3. 执行微调
    result = await instance.fine_tuning.remote(request_data)
    
    # 4. 保存LoRA权重
    await save_lora(result.lora_path)
    
    # 5. 关闭微调实例
    await self._shutdown_instance(instance_id)
    
    return result
```

---

## 第四部分：面试高频考察点

### 4.1 系统设计类问题

#### Q1: 为什么选择Serverless架构？

**考察点**：对Serverless理念的理解

**标准答案**：
1. **资源利用率**：GPU成本高，Serverless可以让多模型共享
2. **弹性伸缩**：根据流量自动扩缩容，scale to 0
3. **成本优化**：只为实际使用的资源付费
4. **运维简化**：用户只需关注模型，无需管理基础设施

#### Q2: 如何实现6-10x的加载加速？

**考察点**：对存储和IO优化的理解

**标准答案**：
1. **存储格式优化**：连续存储tensor，避免随机IO
2. **O_DIRECT I/O**：绕过Page Cache，减少拷贝
3. **Pinned Memory**：支持DMA直接传输到GPU
4. **并行加载**：多线程并行读取

**深挖问题**：
- 为什么O_DIRECT能加速？（减少内核态拷贝）
- Pinned Memory的作用？（支持DMA，避免CPU拷贝）
- 为什么SafeTensors慢？（随机IO + Python开销）

#### Q3: 调度算法如何设计？

**考察点**：对调度系统的理解

**标准答案**：
1. **FCFS**：简单公平，适合无状态场景
2. **Storage-Aware**：考虑数据本地性，减少加载时间
3. **迁移感知**：当资源不足时，考虑迁移现有实例

**深挖问题**：
- 如何估算加载时间？（模型大小 / 带宽）
- 迁移成本如何计算？（KV Cache传输时间 + 重新计算时间）
- 如何避免频繁迁移？（设置迁移阈值）

### 4.2 LLM推理相关问题

#### Q4: vLLM和Transformers后端的区别？

**考察点**：对LLM推理框架的理解

**标准答案**：

| 特性 | Transformers | vLLM |
|------|--------------|------|
| 适用场景 | 小模型、调试 | 大模型、生产 |
| 内存管理 | 简单 | PagedAttention |
| 吞吐量 | 低 | 高 |
| 特性支持 | 完整 | 部分 |

**深挖问题**：
- PagedAttention是什么？（类似操作系统虚拟内存的KV Cache管理）
- vLLM如何优化吞吐？（连续批处理、PagedAttention）

#### Q5: 如何处理KV Cache？

**考察点**：对LLM推理优化的理解

**标准答案**：
1. **Prefix Caching**：缓存公共前缀的KV Cache
2. **KV Cache迁移**：Live Migration时传输KV Cache
3. **内存管理**：使用Pinned Memory池管理

**深挖问题**：
- Prefix Caching的原理？（hash(prompt前缀) → 缓存KV）
- 迁移KV Cache的开销？（与token数成正比）

### 4.3 分布式系统问题

#### Q6: Ray在系统中的作用？

**考察点**：对分布式计算框架的理解

**标准答案**：
1. **Actor模型**：每个组件是Ray Actor
2. **资源管理**：Ray管理GPU/CPU资源分配
3. **任务调度**：Ray调度任务到不同节点
4. **容错**：Ray提供Actor故障恢复

**深挖问题**：
- Ray Actor和普通进程的区别？（有状态、可远程调用）
- 如何保证一致性？（Asyncio Lock）

#### Q7: 如何处理故障？

**考察点**：对系统可靠性的理解

**标准答案**：
1. **实例故障**：Ray Actor自动重启
2. **请求超时**：设置超时机制，返回错误
3. **资源泄漏**：定期检查和清理

---

## 第五部分：算法岗相关的考察点

### 5.1 与LLM应用的结合

#### Q8: 如何支持RAG场景？

**考察点**：对LLM应用场景的理解

**标准答案**：
1. **Embedding模型**：支持部署embedding模型
2. **多模型管理**：同时管理LLM和embedding模型
3. **低延迟**：快速加载不同模型

```python
# 支持embedding模型
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    return await router.encode(request)
```

#### Q9: 如何支持Agent场景？

**考察点**：对Agent系统的理解

**标准答案**：
1. **多模型切换**：Agent可能需要调用不同模型
2. **低延迟**：Agent需要快速响应
3. **资源复用**：多个Agent共享GPU资源

**深挖问题**：
- Agent场景下的调度挑战？（模型切换频繁、延迟敏感）
- 如何优化Agent的响应时间？（预加载常用模型）

#### Q10: LoRA在生产环境的应用？

**考察点**：对模型微调的理解

**标准答案**：
1. **多租户**：一个base model + 多个LoRA适配器
2. **快速切换**：动态加载不同LoRA
3. **资源高效**：LoRA很小，可以缓存很多

```python
# 请求时指定LoRA
{
    "model": "llama-7b",
    "lora_adapter_name": "customer_a_adapter",
    "messages": [...]
}
```

### 5.2 与后训练的结合

#### Q11: Serverless微调的优势？

**考察点**：对训练系统的理解

**标准答案**：
1. **按需分配**：只为微调任务分配GPU
2. **资源共享**：微调和推理共享GPU池
3. **快速启动**：无需预先分配资源

**深挖问题**：
- 微调和推理如何共享GPU？（时间复用）
- 如何保证微调任务的QoS？（优先级调度）

#### Q12: 如何优化微调任务？

**考察点**：对训练优化的理解

**标准答案**：
1. **LoRA**：只训练少量参数
2. **混合精度**：使用float16减少内存
3. **梯度累积**：减少通信开销

### 5.3 与Infra的结合

#### Q13: 如何扩展到多机多卡？

**考察点**：对分布式训练/推理的理解

**标准答案**：
1. **Tensor Parallelism**：模型分片到多卡
2. **Pipeline Parallelism**：模型层分片
3. **Ray调度**：自动分配到不同节点

```python
# vLLM支持Tensor Parallelism
backend_config = {
    "tensor_parallel_size": 4,  # 4卡并行
}
```

#### Q14: 如何优化GPU内存使用？

**考察点**：对GPU内存管理的理解

**标准答案**：
1. **Pinned Memory池**：预分配，避免频繁分配
2. **模型卸载**：LRU策略卸载不常用模型
3. **KV Cache管理**：PagedAttention管理KV Cache

---

## 第六部分：模拟面试Q&A

### 基础问题

**Q: 请介绍一下ServerlessLLM项目？**

> ServerlessLLM是一个支持多模型共享GPU的高性能LLM推理系统，发表在OSDI'24。核心创新有三点：
> 1. 超快模型加载：通过自定义存储格式和O_DIRECT IO，实现6-10x的加载加速
> 2. GPU复用：多模型按需加载到同一GPU，通过智能调度实现资源高效利用
> 3. 统一推理+微调：支持LoRA微调与推理无缝集成
> 
> 系统架构分为三层：User Interface（CLI和API Gateway）、Control Plane（Controller、Scheduler、Store Manager）、Data Plane（Router、Backend、Sllm Store）。使用Ray实现分布式，支持自动扩缩容和Live Migration。

**Q: 你在项目中做了什么贡献？**

> 根据你的实际情况回答，可以包括：
> - 阅读和理解代码
> - 复现和测试
> - 性能分析和优化建议
> - 文档改进

### 进阶问题

**Q: 如果让你优化这个系统，你会怎么做？**

> 1. **调度优化**：使用机器学习预测模型加载时间，而不是简单的带宽计算
> 2. **缓存优化**：实现更智能的缓存策略，如LFU（最不常用）
> 3. **流式支持**：支持Streaming输出，减少TTFT
> 4. **批处理优化**：实现continuous batching提高吞吐

**Q: 这个项目有什么局限性？**

> 1. **仅支持LLM**：不支持其他类型的模型
> 2. **单机为主**：多机支持还在完善
> 3. **冷启动开销**：虽然优化了，但仍有3s+的延迟
> 4. **内存限制**：需要足够的主内存作为缓存

### 开放问题

**Q: ServerlessLLM和vLLM的区别？**

> | 维度 | ServerlessLLM | vLLM |
> |------|---------------|------|
> | 目标 | 多模型共享GPU | 单模型高吞吐 |
> | 优化重点 | 冷启动、资源复用 | 推理吞吐、内存管理 |
> | 适用场景 | 多模型服务 | 单模型大规模服务 |
> | 核心技术 | 快速加载、智能调度 | PagedAttention、Continuous Batching |

**Q: 如何评估这个系统的效果？**

> 1. **TTFT（Time to First Token）**：首token延迟
> 2. **Throughput**：每秒处理的请求数
> 3. **GPU利用率**：GPU实际使用时间占比
> 4. **冷启动时间**：模型从磁盘到可用的时间
> 5. **成本效率**：每请求的GPU成本

---

## 第七部分：如何介绍这个项目

### 7.1 一分钟版本

> ServerlessLLM是我在研究的LLM推理系统，发表在OSDI'24。它解决的核心问题是：如何用有限的GPU资源服务多个LLM模型。
> 
> 传统方案是每个模型独占GPU，成本很高。ServerlessLLM通过三个创新解决了这个问题：
> 1. 超快模型加载：通过优化存储格式，加载速度比传统方案快6-10倍
> 2. GPU复用：多个模型按需加载到同一GPU，通过智能调度实现资源高效利用
> 3. 统一推理+微调：支持LoRA微调与推理无缝集成
> 
> 这个系统特别适合多模型服务场景，比如企业内部有多个不同版本的模型需要部署。

### 7.2 五分钟版本

> **背景**：LLM模型很大，部署成本高。如果每个模型独占GPU，成本会非常高。特别是当模型数量多但每个模型请求量不大时，GPU利用率很低。
> 
> **挑战**：
> 1. 模型加载慢：从磁盘加载7B模型需要20秒
> 2. GPU切换成本高：卸载再加载模型延迟很大
> 3. 资源管理复杂：如何决定哪个模型放在哪个GPU
> 
> **解决方案**：
> 
> **1. 超快模型加载**
> - 问题：传统加载方式是随机IO + Python对象开销
> - 方案：自定义存储格式，连续存储tensor；使用O_DIRECT绕过Page Cache；使用Pinned Memory支持DMA
> - 效果：加载速度提升6-10倍
> 
> **2. GPU复用与智能调度**
> - 问题：多个模型需要共享GPU
> - 方案：实现Storage-Aware调度，考虑数据本地性；支持Live Migration，当资源冲突时迁移模型
> - 效果：1块GPU可以服务10个模型
> 
> **3. 统一推理+微调**
> - 问题：微调和推理需要不同的资源
> - 方案：实现LoRA微调与推理的统一管理；支持按需分配微调资源
> - 效果：一个base model + 100个LoRA适配器
> 
> **系统架构**：
> - 使用Ray实现分布式
> - 三层架构：User Interface、Control Plane、Data Plane
> - 核心组件：Controller、Scheduler、Store Manager、Router、Backend
> 
> **效果**：
> - 模型加载：6-10x加速
> - 资源利用：1 GPU服务10模型
> - 延迟：冷启动3秒内

### 7.3 与算法面试官的对话

**面试官**：你了解LLM推理吗？

> 是的，我研究过ServerlessLLM这个项目，它是一个LLM推理系统。我对LLM推理的流程比较熟悉，包括：
> 1. 模型加载：从磁盘加载到GPU
> 2. 推理执行：prefill和decode阶段
> 3. KV Cache管理：缓存历史token的key-value
> 4. 批处理：continuous batching提高吞吐

**面试官**：KV Cache是什么？

> KV Cache是LLM推理中的重要优化。在生成每个token时，需要计算attention，这需要所有历史token的key和value。如果每次都重新计算，计算量会很大。
> 
> KV Cache的做法是：
> 1. Prefill阶段：计算prompt所有token的key-value，缓存起来
> 2. Decode阶段：只计算新token的key-value，与缓存的拼接
> 
> 这样可以避免重复计算，大幅提高推理速度。

**面试官**：如何优化LLM推理延迟？

> 从几个方面优化：
> 1. **模型加载**：使用ServerlessLLM的快速加载技术
> 2. **KV Cache**：使用Prefix Caching缓存公共前缀
> 3. **批处理**：使用Continuous Batching提高吞吐
> 4. **量化**：使用INT8/INT4减少计算量
> 5. **并行**：使用Tensor Parallelism分布到多卡

---

## 附录：关键代码路径

| 组件 | 文件路径 | 关键函数 |
|------|----------|----------|
| Controller | `sllm/controller.py` | `register()`, `delete()` |
| Router | `sllm/routers/roundrobin_router.py` | `inference()`, `_auto_scaler_loop()` |
| Scheduler | `sllm/schedulers/storage_aware_scheduler.py` | `schedule()`, `get_migration_plans()` |
| Store Manager | `sllm/store_manager.py` | `register()`, `load_to_host()` |
| Backend | `sllm/backends/vllm_backend.py` | `generate()`, `init_backend()` |
| 快速加载 | `sllm_store/torch.py` | `save_dict()`, `load_dict()` |

---

## 总结

ServerlessLLM是一个很有价值的项目，它展示了如何在资源受限的环境下高效服务多个LLM模型。通过学习这个项目，你可以：

1. **理解LLM推理的全貌**：从模型加载到推理执行
2. **掌握系统优化技巧**：IO优化、内存管理、调度算法
3. **了解分布式系统设计**：Ray、Actor模型、一致性
4. **为面试准备素材**：系统设计、性能优化、问题解决

在面试中，要能够：
- 清晰地介绍项目背景和解决的问题
- 深入解释核心技术原理
- 讨论设计决策和权衡
- 提出改进建议

祝你面试顺利！
