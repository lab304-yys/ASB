# 使用 HuggingFace 模型指南 / HuggingFace Models Guide

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 中文说明

### 概述

ASB 已经支持使用 HuggingFace 上的开源模型进行测试。系统会自动从 HuggingFace Hub 下载并加载模型。

### 支持的模型类型

ASB 支持三种类型的 LLM：

1. **API 模型**（GPT、Claude、Gemini）- 通过 API 调用
2. **Ollama 模型** - 通过 Ollama 后端运行，使用 `ollama/` 前缀
3. **HuggingFace 模型** - 直接从 HuggingFace Hub 加载，不需要前缀

### 如何使用 HuggingFace 模型

#### 1. 配置环境

如果您想使用需要授权的 HuggingFace 模型（如 Meta 的 Llama 模型），需要设置 HuggingFace 访问令牌：

```bash
# 创建或编辑 .env 文件
echo "HF_AUTH_TOKENS=your_huggingface_token_here" >> .env
```

获取 HuggingFace 令牌的步骤：
1. 访问 https://huggingface.co/settings/tokens
2. 创建一个新的访问令牌
3. 将令牌添加到 `.env` 文件

#### 2. 在配置文件中指定模型

在 YAML 配置文件中，直接使用 HuggingFace 模型的完整路径：

```yaml
llms:
  # HuggingFace 模型 - 使用模型的完整路径
  - meta-llama/Llama-2-7b-chat-hf
  - mistralai/Mistral-7B-Instruct-v0.2
  - google/gemma-2-9b-it
  - microsoft/Phi-3-mini-4k-instruct
  
  # 也可以混合使用不同类型的模型
  - gpt-4o-mini                    # API 模型
  - ollama/llama3:8b               # Ollama 模型
  - meta-llama/Llama-2-13b-chat-hf # HuggingFace 模型
```

#### 3. 运行测试

使用标准的运行命令：

```bash
python scripts/agent_attack.py --cfg_path config/your_config.yml
```

### 示例配置文件

我们提供了一个示例配置文件 `config/huggingface_example.yml`，展示如何使用 HuggingFace 模型：

```bash
python scripts/agent_attack.py --cfg_path config/huggingface_example.yml
```

### 推荐的 HuggingFace 模型

以下是一些推荐的开源模型，可以直接在 ASB 中使用：

#### 小型模型（适合 GPU 内存有限的情况）
- `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- `google/gemma-2-2b-it` (2B)
- `Qwen/Qwen2-7B-Instruct` (7B)

#### 中型模型（推荐用于平衡性能和资源）
- `meta-llama/Llama-2-7b-chat-hf` (7B)
- `mistralai/Mistral-7B-Instruct-v0.2` (7B)
- `google/gemma-2-9b-it` (9B)

#### 大型模型（需要更多 GPU 内存）
- `meta-llama/Llama-2-13b-chat-hf` (13B)
- `meta-llama/Meta-Llama-3-70B-Instruct` (70B) - 需要多 GPU 或量化

### 系统要求

- **GPU 内存**：
  - 7B 模型：至少 16GB VRAM（推荐 24GB）
  - 13B 模型：至少 24GB VRAM（推荐 32GB）
  - 70B 模型：需要多 GPU 或量化技术

- **RAM**：至少 32GB 系统内存

### 故障排除

#### 问题 1：CUDA 内存不足

**解决方案**：使用较小的模型或启用模型量化。

#### 问题 2：模型下载失败

**解决方案**：
- 检查网络连接
- 确保 HuggingFace 令牌有效（如果需要）
- 尝试使用 HuggingFace 镜像源

#### 问题 3：权限错误

**解决方案**：某些模型（如 Llama）需要：
1. 在 HuggingFace 上接受模型的许可协议
2. 使用有效的访问令牌

### 与 Ollama 的区别

| 特性 | Ollama | HuggingFace | vLLM (高级) |
|------|---------|-------------|------------|
| 模型格式 | 使用 `ollama/model:tag` | 使用 `username/model-name` | 使用 `username/model-name` |
| 安装要求 | 需要安装 Ollama | 仅需 transformers 库 | 需要安装 vllm 库 |
| 模型管理 | 通过 `ollama pull` | 自动下载 | 自动下载 |
| GPU 使用 | Ollama 管理 | 由 transformers 自动管理 | vLLM 优化推理 |
| 内存优化 | Ollama 优化 | 标准 PyTorch 加载 | 高度优化（PagedAttention） |
| 推理速度 | 中等 | 标准 | 最快 |
| 使用场景 | 简单部署 | 标准使用 | 生产环境/高吞吐 |

**注意**: 如果您需要使用 vLLM 后端以获得更快的推理速度，请在命令行中指定 `--use_backend vllm`。

---

<a name="english"></a>
## English

### Overview

ASB already supports using open-source models from HuggingFace for testing. The system will automatically download and load models from the HuggingFace Hub.

### Supported Model Types

ASB supports three types of LLMs:

1. **API Models** (GPT, Claude, Gemini) - Called via API
2. **Ollama Models** - Run through Ollama backend, use `ollama/` prefix
3. **HuggingFace Models** - Loaded directly from HuggingFace Hub, no prefix needed

### How to Use HuggingFace Models

#### 1. Configure Environment

If you want to use gated HuggingFace models (like Meta's Llama models), you need to set up a HuggingFace access token:

```bash
# Create or edit .env file
echo "HF_AUTH_TOKENS=your_huggingface_token_here" >> .env
```

Steps to get a HuggingFace token:
1. Visit https://huggingface.co/settings/tokens
2. Create a new access token
3. Add the token to your `.env` file

#### 2. Specify Models in Configuration

In your YAML configuration file, use the full HuggingFace model path:

```yaml
llms:
  # HuggingFace models - use the full model path
  - meta-llama/Llama-2-7b-chat-hf
  - mistralai/Mistral-7B-Instruct-v0.2
  - google/gemma-2-9b-it
  - microsoft/Phi-3-mini-4k-instruct
  
  # You can also mix different types of models
  - gpt-4o-mini                    # API model
  - ollama/llama3:8b               # Ollama model
  - meta-llama/Llama-2-13b-chat-hf # HuggingFace model
```

#### 3. Run Tests

Use the standard run command:

```bash
python scripts/agent_attack.py --cfg_path config/your_config.yml
```

### Example Configuration

We provide an example configuration file `config/huggingface_example.yml` that shows how to use HuggingFace models:

```bash
python scripts/agent_attack.py --cfg_path config/huggingface_example.yml
```

### Recommended HuggingFace Models

Here are some recommended open-source models that can be used directly in ASB:

#### Small Models (suitable for limited GPU memory)
- `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- `google/gemma-2-2b-it` (2B)
- `Qwen/Qwen2-7B-Instruct` (7B)

#### Medium Models (recommended for balanced performance and resources)
- `meta-llama/Llama-2-7b-chat-hf` (7B)
- `mistralai/Mistral-7B-Instruct-v0.2` (7B)
- `google/gemma-2-9b-it` (9B)

#### Large Models (require more GPU memory)
- `meta-llama/Llama-2-13b-chat-hf` (13B)
- `meta-llama/Meta-Llama-3-70B-Instruct` (70B) - Requires multi-GPU or quantization

### System Requirements

- **GPU Memory**:
  - 7B models: At least 16GB VRAM (24GB recommended)
  - 13B models: At least 24GB VRAM (32GB recommended)
  - 70B models: Requires multi-GPU or quantization techniques

- **RAM**: At least 32GB system memory

### Troubleshooting

#### Issue 1: CUDA Out of Memory

**Solution**: Use smaller models or enable model quantization.

#### Issue 2: Model Download Fails

**Solution**:
- Check network connection
- Ensure HuggingFace token is valid (if required)
- Try using a HuggingFace mirror

#### Issue 3: Permission Error

**Solution**: Some models (like Llama) require:
1. Accepting the model's license agreement on HuggingFace
2. Using a valid access token

### Differences from Ollama

| Feature | Ollama | HuggingFace | vLLM (Advanced) |
|---------|---------|-------------|-----------------|
| Model Format | Use `ollama/model:tag` | Use `username/model-name` | Use `username/model-name` |
| Installation | Requires Ollama installation | Only needs transformers library | Requires vllm library |
| Model Management | Via `ollama pull` | Automatic download | Automatic download |
| GPU Usage | Managed by Ollama | Auto-managed by transformers | vLLM optimized inference |
| Memory Optimization | Ollama optimized | Standard PyTorch loading | Highly optimized (PagedAttention) |
| Inference Speed | Medium | Standard | Fastest |
| Use Case | Simple deployment | Standard usage | Production/High throughput |

**Note**: If you need to use vLLM backend for faster inference, specify `--use_backend vllm` in the command line.
