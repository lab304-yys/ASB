# HuggingFace Model Integration Summary / HuggingFace 模型集成总结

## 简介 / Overview

**中文**: ASB 代码库已经完全支持使用 HuggingFace 模型进行测试。您无需修改任何核心代码，只需在配置文件中指定 HuggingFace 模型名称即可。

**English**: The ASB codebase already fully supports using HuggingFace models for testing. You don't need to modify any core code - just specify HuggingFace model names in your configuration files.

## 快速开始 / Quick Start

### 1. 在配置文件中添加 HuggingFace 模型 / Add HuggingFace Models to Config

编辑任何配置文件（如 `config/DPI.yml`）：
Edit any config file (e.g., `config/DPI.yml`):

```yaml
llms:
  - microsoft/Phi-3-mini-4k-instruct     # HuggingFace 模型
  - gpt-4o-mini                          # API 模型
  - ollama/llama3:8b                     # Ollama 模型
```

### 2. (可选) 设置 HuggingFace Token / (Optional) Set HuggingFace Token

对于需要授权的模型（如 Llama），添加到 `.env` 文件：
For gated models (like Llama), add to `.env` file:

```bash
HF_AUTH_TOKENS=your_huggingface_token_here
```

### 3. 运行测试 / Run Tests

```bash
python scripts/agent_attack.py --cfg_path config/huggingface_example.yml
```

## 工作原理 / How It Works

系统会自动检测模型类型：
The system automatically detects model types:

1. **API 模型**: 以 `gpt`, `claude`, `gemini` 开头
   **API Models**: Start with `gpt`, `claude`, `gemini`
   
2. **Ollama 模型**: 以 `ollama/` 开头
   **Ollama Models**: Start with `ollama/`
   
3. **HuggingFace 模型**: 其他所有模型（通常包含 `/`）
   **HuggingFace Models**: All other models (usually contain `/`)

## 推荐模型 / Recommended Models

### 小型模型 / Small Models (< 16GB VRAM)
- `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- `google/gemma-2-2b-it` (2B)

### 中型模型 / Medium Models (16-24GB VRAM)
- `mistralai/Mistral-7B-Instruct-v0.2` (7B)
- `Qwen/Qwen2-7B-Instruct` (7B)

### 大型模型 / Large Models (> 24GB VRAM)
- `meta-llama/Llama-2-13b-chat-hf` (13B) - 需要 HF token

## 文档位置 / Documentation Locations

- **详细指南 / Detailed Guide**: `docs/HUGGINGFACE_MODELS.md`
- **示例配置 / Example Config**: `config/huggingface_example.yml`
- **主 README**: 已更新，包含 HuggingFace 部分 / Updated with HuggingFace section

## 技术细节 / Technical Details

### 修改的文件 / Modified Files

1. **docs/HUGGINGFACE_MODELS.md** - 新增，双语文档
2. **config/huggingface_example.yml** - 新增，示例配置
3. **README.md** - 更新，添加 HuggingFace 说明
4. **scripts/agent_attack.py** - 更新，修复模型名称解析
5. **scripts/agent_attack_pot.py** - 更新，修复模型名称解析

### 核心逻辑 / Core Logic

系统使用 `HfNativeLLM` 类处理 HuggingFace 模型：
The system uses `HfNativeLLM` class for HuggingFace models:

```python
# In aios/llm_core/llms.py
if use_backend == "ollama" or llm_name.startswith("ollama"):
    self.model = OllamaLLM(...)
elif use_backend == "vllm":
    self.model = vLLM(...)
else:
    # 使用 HuggingFace 模型 / Use HuggingFace models
    self.model = HfNativeLLM(...)
```

## 测试结果 / Test Results

✓ 所有模型检测测试通过 / All model detection tests passed
✓ 配置文件加载成功 / Configuration file loading successful
✓ 模块导入正常 / Module imports working
✓ 安全扫描通过 / Security scan passed (0 alerts)

## 常见问题 / FAQ

**Q: 我需要安装额外的软件吗？**
**Q: Do I need to install additional software?**

A: 不需要。只需要已有的 Python 依赖（transformers, torch 等）。
A: No. Only existing Python dependencies are needed (transformers, torch, etc.).

**Q: HuggingFace 模型会自动下载吗？**
**Q: Will HuggingFace models download automatically?**

A: 是的。首次使用时会自动从 HuggingFace Hub 下载。
A: Yes. Models will be automatically downloaded from HuggingFace Hub on first use.

**Q: 可以混合使用不同类型的模型吗？**
**Q: Can I mix different types of models?**

A: 可以！您可以在同一个配置文件中混合使用 API、Ollama 和 HuggingFace 模型。
A: Yes! You can mix API, Ollama, and HuggingFace models in the same configuration.

## 支持 / Support

如有问题，请参考：
For issues, please refer to:

- 详细文档: `docs/HUGGINGFACE_MODELS.md`
- GitHub Issues: https://github.com/lab304-yys/ASB/issues
