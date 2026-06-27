# ServerlessLLM 学习增量记录

> 本文档记录在复现过程中对比前两轮学习新发现的点
> 持续更新，每次学习后记录新的收获

---

## 记录格式说明

每次记录包含：
- **日期**：学习日期
- **阶段**：复现阶段编号
- **新发现**：之前不知道的知识点
- **深入理解**：对之前知识点的更深入理解
- **实践收获**：动手实践中获得的经验
- **问题与解决**：遇到的问题和解决方案

---

## 第一轮增量：环境搭建与基础功能

### 日期：复现 Day 1-3

#### 新发现1：Ray的资源管理机制

**之前理解**：Ray只是一个分布式计算框架

**新发现**：
```python
# Ray通过resources字段管理资源
ray start --head --resources='{"control_node": 1}'

# Worker节点的资源声明
ray start --address=head:6379 --resources='{"worker_node": 1, "worker_id_0": 1}'

# 代码中通过options指定资源需求
@ray.remote(resources={"worker_node": 0.1, "worker_id_0": 0.1})
def my_task():
    pass
```

**深入理解**：
- Ray的资源是**逻辑概念**，不是物理限制
- `resources={"worker_node": 0.1}`表示任务需要0.1单位的worker_node资源
- 这是一种**调度提示**，告诉Ray这个任务应该调度到哪个节点
- 实际资源使用由CUDA/OS管理，Ray只负责调度

#### 新发现2：Docker Compose的设备映射

**之前理解**：Docker只是容器化工具

**新发现**：
```yaml
# docker-compose.yml中的GPU配置
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: ["gpu"]
          device_ids: ["0"]  # 精确指定GPU 0
```

**深入理解**：
- `device_ids`可以精确控制容器使用哪个GPU
- 这对于ServerlessLLM很重要：每个Worker容器绑定一个GPU
- 通过Docker Compose可以模拟多机环境（每个容器模拟一个节点）

#### 新发现3：sllm-store的模型格式

**之前理解**：sllm-store只是模型存储工具

**新发现**：
```python
# sllm-store save的内部流程
1. 加载HuggingFace模型
2. 提取state_dict
3. 调用save_tensors()将tensor保存为连续二进制
4. 生成tensor_index.json（包含offset, size, shape, stride, dtype）
5. 保存tokenizer和配置文件
```

**深入理解**：
- sllm-store格式是**自定义的二进制格式**
- tensor_index.json是关键：它记录了每个tensor在文件中的精确位置
- 这种格式支持O_DIRECT I/O，因为数据是连续存储的
- 与SafeTensors的区别：SafeTensors每个tensor有独立的header，导致随机IO

#### 实践收获：模型转换的注意事项

```bash
# 1. 需要先下载模型到本地
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /tmp/qwen7b

# 2. sllm-store save需要足够的磁盘空间
# 原始模型 + 转换后的模型 = 2倍空间

# 3. 转换时间与模型大小成正比
# 7B模型约需要2-5分钟
```

#### 问题与解决

**问题1**：Docker构建失败，提示CUDA版本不匹配
```
原因：本地CUDA版本与Dockerfile中的CUDA版本不一致
解决：修改Dockerfile中的CUDA_VERSION参数，或升级本地CUDA
```

**问题2**：sllm-store save失败，提示内存不足
```
原因：模型太大，系统内存不足
解决：减小mem_pool_size，或使用更大的内存机器
```

---

## 第二轮增量：性能测试与调度

### 日期：复现 Day 4-7

#### 新发现1：Pinned Memory Pool的LRU驱逐机制

**之前理解**：Pinned Memory只是一个缓存池

**新发现**：
```cpp
// checkpoint_store.cpp中的LRU驱逐
int CheckpointStore::LoadModelFromDisk(const std::string& model_path) {
    // 1. 尝试分配内存
    int remaining_size = model->AllocatePinnedMemory(memory_pool_);
    
    if (remaining_size > 0) {
        // 2. 内存不足，需要驱逐
        // 获取所有模型的最后访问时间
        std::sort(model_last_access_time.begin(), model_last_access_time.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // 3. 按LRU顺序驱逐
        for (const auto& [model_path, last_access_time] : model_last_access_time) {
            int freed_chunks = model->TryFreeHost();  // 非阻塞释放
            mem_chunks_needed -= freed_chunks;
            if (mem_chunks_needed <= 0) break;
        }
    }
    
    // 4. 重新分配内存
    model->AllocatePinnedMemory(memory_pool_);
    
    // 5. 从磁盘加载到内存
    model->ToHost(num_thread_);
}
```

**深入理解**：
- LRU驱逐是**非阻塞**的（TryFreeHost）
- 驱逐的单位是**chunk**，不是整个模型
- 这保证了加载过程不会被阻塞太久
- 但可能导致**驱逐风暴**：频繁驱逐导致性能下降

#### 新发现2：Storage-Aware调度的延迟估算

**之前理解**：调度只是选择空闲节点

**新发现**：
```python
def _get_model_loading_time(self, model_name, model_size, hardware_info,
                            node_waiting_time, pinned_memory_pool):
    latency = 0
    
    if model_name in pinned_memory_pool:
        # 模型已在内存中，用PCIe带宽
        latency += model_size / hardware_info["pcie_bandwidth"]  # ~25GB/s
    else:
        # 模型在磁盘上，用磁盘带宽
        latency += node_waiting_time + model_size / hardware_info["disk_bandwidth"]  # ~3GB/s
    
    return latency
```

**深入理解**：
- 调度决策基于**延迟估算**，不是简单的负载均衡
- 两个关键因素：数据本地性 + 带宽
- PCIe带宽（25GB/s）远大于磁盘带宽（3GB/s）
- 这解释了为什么Pinned Memory Pool如此重要

#### 新发现3：Live Migration的增量传输

**之前理解**：迁移是全量传输KV Cache

**新发现**：
```python
# migration_router.py中的增量迁移
async def execute_migration_plan(self, migration_plan):
    # 1. 在目标节点创建新实例
    new_instance = await self._create_instance(target_node)
    
    # 2. 增量迁移KV Cache
    n_previous_tokens = 0
    while True:
        current_tokens = await source.get_current_tokens()
        n_current_tokens = len(current_tokens[0])
        n_delta_tokens = n_current_tokens - n_previous_tokens
        
        # 3. 当增量很小时，完成迁移
        if n_delta_tokens <= self.migration_delta:  # 默认20 tokens
            break
        
        # 4. 只传输增量部分
        await new.resume_kv_cache(current_tokens)
        n_previous_tokens = n_current_tokens
    
    # 5. 原子切换
    await self._atomic_switch(source, new)
```

**深入理解**：
- 迁移是**多轮迭代**的
- 每轮只传输**增量tokens**
- `migration_delta`控制迁移粒度（默认20 tokens）
- 最后一轮需要**原子切换**，保证请求不丢失

#### 实践收获：性能测试的关键细节

```python
# 1. CUDA预热很重要
def _warmup_cuda():
    for i in range(num_gpus):
        torch.ones(1).to(f"cuda:{i}")
        torch.cuda.synchronize()  # 必须同步！

# 2. 内存清理必须彻底
def cleanup():
    del model
    gc.collect()  # Python垃圾回收
    torch.cuda.empty_cache()  # PyTorch缓存
    torch.cuda.synchronize()  # 等待GPU操作完成

# 3. 测量时间要排除首次开销
# 首次加载可能有JIT编译、CUDA初始化等开销
```

#### 问题与解决

**问题3**：Live Migration触发失败
```
原因：migration_delta设置太小，导致迁移无法完成
解决：增大migration_delta（从20改为50）
```

**问题4**：调度器日志显示"No available node"
```
原因：所有节点的GPU都被占用了
解决：
1. 增加节点数
2. 减少每个模型的GPU需求
3. 启用自动扩缩容，让空闲模型释放GPU
```

---

## 第三轮增量：高级功能与源码分析

### 日期：复现 Day 8-17

#### 新发现1：vLLM的PagedAttention

**之前理解**：vLLM只是更快的推理引擎

**新发现**：
```python
# vllm_backend.py中的配置
filtered_engine_config["enable_prefix_caching"] = True  # 默认启用

# Prefix Caching的工作原理：
# 1. 对prompt的每个token计算hash
# 2. 如果hash命中缓存，直接复用KV Cache
# 3. 避免重复计算，大幅减少TTFT
```

**深入理解**：
- PagedAttention将KV Cache分成固定大小的**page**
- 类似操作系统的虚拟内存管理
- Prefix Caching是PagedAttention的扩展
- 对于共享前缀的请求（如system prompt），效果显著

#### 新发现2：LoRA的动态加载机制

**之前理解**：LoRA只是在模型旁边加一个小网络

**新发现**：
```python
# transformers_backend.py中的LoRA加载
def load_lora_adapter(self, lora_name, lora_path):
    # 1. 检查是否已加载
    if hasattr(self.model, 'peft_config') and lora_name in self.model.peft_config:
        return  # 已加载，直接返回
    
    # 2. 使用sllm_store加载LoRA
    self.model = load_lora(
        self.model, lora_name, lora_path,
        device_map=device_map,
        storage_path=storage_path,
        torch_dtype=torch_dtype
    )

# 推理时指定LoRA
def generate(self, request_data):
    lora_adapter_name = request_data.get("lora_adapter_name")
    if lora_adapter_name:
        generate_kwargs["adapter_names"] = [lora_adapter_name]
    
    outputs = self.model.generate(**inputs, **generate_kwargs)
```

**深入理解**：
- LoRA是**动态加载**的，不是预先合并的
- 一个base model可以同时加载**多个LoRA**
- 推理时通过`adapter_names`指定使用哪个LoRA
- 这实现了"一个base model + 100个LoRA"的架构

#### 新发现3：自动扩缩容的决策逻辑

**之前理解**：扩缩容只是简单的阈值判断

**新发现**：
```python
# roundrobin_router.py中的扩缩容逻辑
async def auto_scaler(auto_scaling_metrics, auto_scaling_config):
    request_count = auto_scaling_metrics.get("request_count", 0)
    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 2)
    
    # 计算期望实例数
    desired_instances = (request_count + target_ongoing_requests - 1) // target_ongoing_requests
    
    # 限制在min/max范围内
    desired_instances = min(max_instances, max(min_instances, desired_instances))
    
    return desired_instances

# 缩容时的keep_alive机制
if desired_instances < num_running_instances:
    keep_alive = auto_scaling_config.get("keep_alive", 0)
    if self.idle_time >= keep_alive:
        await self._stop_instance()
    else:
        self.idle_time += self.loop_interval
```

**深入理解**：
- 扩缩容基于**请求队列长度**，不是CPU/GPU使用率
- `target_ongoing_requests`是关键参数：每个实例处理的请求数
- `keep_alive`防止频繁扩缩容（抖动）
- 可以缩容到0（scale to zero），节省资源

#### 实践收获：源码阅读技巧

```python
# 1. 从入口点开始
# sllm/cli/clic.py → deploy_model() → Controller.register()

# 2. 跟踪数据流
# 请求 → API Gateway → Router → Backend → 返回

# 3. 关注异步代码
# asyncio.create_task() 创建后台任务
# await 等待异步结果
# async with lock 获取锁

# 4. 理解Ray Actor
# @ray.remote 定义Actor
# .remote() 异步调用
# ray.get() 同步等待结果
```

#### 问题与解决

**问题5**：vLLM后端启动失败
```
原因：vLLM版本与CUDA版本不兼容
解决：使用项目推荐的vLLM版本（0.11.2）
```

**问题6**：LoRA加载后推理结果异常
```
原因：LoRA的dtype与base model不匹配
解决：确保LoRA和base model使用相同的torch_dtype
```

---

## 第四轮增量：压力测试与优化

### 日期：复现 Day 18-20

#### 新发现1：系统的并发限制

**之前理解**：系统可以处理无限并发

**新发现**：
```python
# 并发限制来自多个层面：

# 1. Router的请求队列
self.request_queue = asyncio.Queue()

# 2. Instance的并发限制
class InstanceHandle:
    max_queue_length: int  # 每个实例的最大队列长度
    
    async def add_requests(self, num_requests):
        if self.concurrency + num_requests > self.max_queue_length:
            return False  # 队列满了，拒绝请求

# 3. Ray的资源限制
@ray.remote(num_gpus=1)  # 每个Actor占用1个GPU
```

**深入理解**：
- 并发限制是**多层**的：Router → Instance → Ray
- `max_queue_length`控制每个实例的并发数
- 队列满时会**拒绝请求**，不是无限排队
- 需要根据GPU显存和模型大小调整`max_queue_length`

#### 新发现2：性能瓶颈的类型

**之前理解**：性能瓶颈只是GPU使用率高

**新发现**：
```python
# 瓶颈类型：

# 1. GPU瓶颈
# 症状：GPU利用率100%，显存接近满
# 原因：模型太大，或批处理太大
# 优化：量化、减小批处理大小

# 2. IO瓶颈
# 症状：磁盘利用率100%，iowait高
# 原因：模型加载频繁，Pinned Memory Pool太小
# 优化：增大Pinned Memory Pool，优化存储格式

# 3. CPU瓶颈
# 症状：CPU利用率100%，但GPU利用率低
# 原因：Python GIL，或调度逻辑太复杂
# 优化：使用多进程，优化调度算法

# 4. 网络瓶颈
# 症状：网络延迟高，带宽打满
# 原因：模型分布在多机，需要跨机加载
# 优化：优化数据本地性，使用更快的网络
```

**深入理解**：
- 瓶颈分析需要**多维度**观察
- 不同瓶颈有不同的**症状**和**优化方法**
- 优化时要**先找最大瓶颈**，不要盲目优化

#### 新发现3：系统监控的关键指标

**之前理解**：监控只是看日志

**新发现**：
```python
# 关键监控指标：

# 1. 性能指标
ttft = time_to_first_token  # 首token延迟
throughput = requests_per_second  # 吞吐量
latency_p99 = percentile(latencies, 99)  # P99延迟

# 2. 资源指标
gpu_utilization = nvidia-smi  # GPU利用率
gpu_memory_used = nvidia-smi  # 显存使用
cpu_utilization = top  # CPU利用率
disk_io = iostat  # 磁盘IO

# 3. 业务指标
request_count = total_requests  # 请求数
error_rate = failed_requests / total_requests  # 错误率
model_switches = count_model_switches()  # 模型切换次数
```

**深入理解**：
- 监控需要**分层**：性能、资源、业务
- 不同指标反映**不同问题**
- 需要建立**基线**，才能发现异常

#### 实践收获：优化策略

```python
# 1. 优化加载速度
# - 增大Pinned Memory Pool
# - 使用NVMe SSD
# - 减少模型切换

# 2. 优化推理速度
# - 使用vLLM后端
# - 启用Prefix Caching
# - 使用量化模型

# 3. 优化资源使用
# - 启用自动扩缩容
# - 调整max_queue_length
# - 使用scale to zero
```

#### 问题与解决

**问题7**：压力测试时请求超时
```
原因：请求队列满了，新请求被拒绝
解决：
1. 增大max_queue_length
2. 增加模型实例数
3. 优化单个请求的处理时间
```

**问题8**：GPU显存泄漏
```
原因：模型卸载不彻底，显存没有完全释放
解决：
1. 调用torch.cuda.empty_cache()
2. 检查是否有引用未释放
3. 使用nvidia-smi监控显存
```

---

## 总结：关键收获

### 技术层面

1. **Ray的资源管理**是逻辑的，不是物理的
2. **Pinned Memory Pool**是性能关键，LRU驱逐是非阻塞的
3. **Storage-Aware调度**基于延迟估算，不是简单负载均衡
4. **Live Migration**是增量传输，不是全量传输
5. **LoRA**是动态加载的，一个base model可以服务多个LoRA
6. **自动扩缩容**基于请求队列长度，支持scale to zero
7. **性能瓶颈**需要多维度分析，不同瓶颈不同优化

### 实践层面

1. **环境准备**很重要，CUDA/Docker/依赖版本要匹配
2. **模型转换**需要足够的磁盘空间
3. **性能测试**需要预热和清理
4. **源码阅读**要从入口点开始，跟踪数据流
5. **监控**要分层：性能、资源、业务

### 面试准备

1. **能够解释**每个组件的作用和原理
2. **能够分析**性能瓶颈和优化方法
3. **能够设计**验证实验和压力测试
4. **能够讨论**适用场景和局限性
5. **能够提出**改进方向和优化建议

---

## 更新日志

| 日期 | 阶段 | 主要收获 |
|------|------|----------|
| Day 1-3 | 基础功能 | Ray资源管理、Docker设备映射、sllm-store格式 |
| Day 4-7 | 性能测试 | LRU驱逐、延迟估算、增量迁移 |
| Day 8-17 | 高级功能 | PagedAttention、LoRA动态加载、自动扩缩容 |
| Day 18-20 | 压力测试 | 并发限制、瓶颈分析、监控指标 |

---

*持续更新中...*
