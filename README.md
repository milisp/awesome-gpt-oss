# Awesome GPT-OSS [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of awesome GPT-OSS resources, tools, tutorials, and projects. OpenAI's first fully open-source language model family since GPT-2.


GPT-OSS represents OpenAI's return to open-source AI development with two powerful reasoning models: **gpt-oss-120b** and **gpt-oss-20b**. Released under the Apache 2.0 license, these models deliver state-of-the-art performance with configurable reasoning effort, full chain-of-thought access, and native tool use capabilities.

## 📋 Contents

- [Official Resources](#official-resources)
- [Models](#models)
- [Inference Engines](#inference-engines)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Development Tools](#development-tools)
- [Integrations](#integrations)
- [Fine-tuning](#fine-tuning)
- [Applications](#applications)
- [Tutorials](#tutorials)
- [Research](#research)
- [Safety](#safety)
- [Community](#community)

## 🏢 Official Resources

- [OpenAI GPT-OSS Announcement](https://openai.com/index/introducing-gpt-oss/) - Official release announcement
- [GPT-OSS GitHub Repository](https://github.com/openai/gpt-oss) - Official implementation and reference code
- [GPT-OSS Model Card](https://openai.com/index/gpt-oss-model-card/) - Comprehensive model documentation
- [Open Models Page](https://openai.com/open-models/) - OpenAI's dedicated open models page
- [OpenAI Harmony](https://github.com/openai/harmony) - Response format library for GPT-OSS
- [Try gpt-oss](https://gpt-oss.com/) - gpt-oss playground

## 🤖 Models

### Hugging Face Models

- [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) - 120B parameter model (117B total, 5.1B active)
- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) - 20B parameter model (21B total, 3.6B active)

### Model Specifications

| Model | Parameters | Active Parameters | Memory Requirement | Hardware |
|-------|------------|-------------------|-------------------|----------|
| gpt-oss-120b | 117B | 5.1B | 80GB | Single H100 |
| gpt-oss-20b | 21B | 3.6B | 16GB | Consumer GPU |

### Key Features

- **Apache 2.0 License** - Permissive open-source licensing
- **MXFP4 Quantization** - Native 4-bit quantization for efficient inference
- **Mixture of Experts (MoE)** - Optimized for performance and efficiency
- **Configurable Reasoning** - Adjustable effort levels (low, medium, high)
- **Full Chain-of-Thought** - Complete access to reasoning process
- **Tool Use Capabilities** - Web browsing, Python execution, function calling

## 🚀 Inference Engines

### vLLM
- [vLLM GPT-OSS Support](https://docs.vllm.ai/en/latest/models/supported_models.html) - Official vLLM implementation
- [Flash Attention 3 Kernels](https://github.com/kernels-community/vllm-flash-attn3) - Optimized attention kernels for Hopper GPUs
- Installation: `pip install --pre vllm==0.10.1+gptoss`

### Ollama
- [Ollama GPT-OSS Models](https://ollama.com/library/gpt-oss) - Easy local deployment
- [OpenAI Cookbook - Ollama Guide](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama) - Official tutorial
- Quick start: `ollama pull gpt-oss:20b && ollama run gpt-oss:20b`

### llama.cpp
- [llama.cpp GPT-OSS Support](https://github.com/ggerganov/llama.cpp) - CPU and GPU inference
- [GGUF Models](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) - Quantized models for llama.cpp

### Transformers
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/gpt_oss) - Official integration
- [Transformers Serve](https://huggingface.co/docs/transformers/main/en/llm_tutorial#transformers-serve) - OpenAI-compatible server

## 💻 Local Deployment

### Consumer Hardware
- [LM Studio](https://lmstudio.ai/) - User-friendly desktop application
- [Jan](https://jan.ai/) - Open-source ChatGPT alternative
- [Msty](https://msty.app/) - Multi-platform LLM client
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - Desktop client with Ollama support

### Enterprise Hardware
- [NVIDIA RTX Optimization](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/) - RTX-optimized deployment
- [Apple Metal Implementation](https://github.com/openai/gpt-oss/tree/main/gpt_oss/metal) - Native Metal support for Apple Silicon
- [AMD ROCm Support](https://huggingface.co/blog/welcome-openai-gpt-oss#amd-instinct-support) - AMD GPU compatibility

## ☁️ Cloud Deployment

### Major Cloud Providers
- [Azure AI Foundry](https://azure.microsoft.com/en-us/blog/openais-open%E2%80%91source-model-gpt%E2%80%91oss-on-azure-ai-foundry-and-windows-ai-foundry/) - Microsoft's AI platform
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/guides/gpt-oss) - Multi-provider access
- [AWS SageMaker](https://aws.amazon.com/sagemaker/) - Amazon's ML platform
- [Northflank](https://northflank.com/blog/self-host-openai-gpt-oss-120b-open-source-chatgpt) - GPU-optimized deployment
- [Fireworks AI](https://fireworks.ai/) - High-performance inference
- [Cerebras](https://cerebras.ai/) - Ultra-fast inference (2-4k tokens/sec)

### Edge Computing
- [Microsoft AI Foundry Local](https://docs.microsoft.com/en-us/azure/ai-studio/) - On-device inference for Windows
- [Ollama Turbo](https://ollama.com/turbo) - Hosted Ollama service for large models

## 🛠️ Development Tools

### Python Libraries
- [gpt-oss](https://pypi.org/project/gpt-oss/) - Official Python package
- [OpenAI Python SDK](https://github.com/openai/openai-python) - Compatible with local endpoints
- [LangChain](https://python.langchain.com/docs/integrations/llms/ollama) - LLM application framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API across providers

### JavaScript/TypeScript
- [Responses.js](https://github.com/huggingface/responses.js) - Response API client library
- [Vercel AI SDK](https://sdk.vercel.ai/docs) - React/Next.js integration
- [OpenAI JS SDK](https://github.com/openai/openai-node) - Node.js client

### APIs and Protocols
- [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) - Compatible with OpenAI format
- [Responses API](https://platform.openai.com/docs/api-reference/responses) - Advanced streaming interface
- [OpenAI Harmony Format](https://cookbook.openai.com/articles/gpt-oss/harmony-response-format) - New response format

## 🔗 Integrations

### Chat Interfaces
- [Open WebUI](https://github.com/open-webui/open-webui) - Feature-rich web interface
- [ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - Self-hosted ChatGPT UI
- [LibreChat](https://github.com/danny-avila/LibreChat) - Multi-model chat platform
- [LobeChat](https://github.com/lobehub/lobe-chat) - Modern chat interface

### IDE Extensions
- [Continue](https://github.com/continuedev/continue) - Open-source AI code assistant
- [AI Toolkit for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio) - Microsoft's official VSCode extension
- [CodeGPT](https://github.com/carlrobertoh/CodeGPT) - IntelliJ plugin

### Agent Frameworks
- [OpenAI Agents SDK](https://github.com/openai/agents) - Official agent development framework
- [AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversation framework
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Role-playing AI agents
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow orchestration

## 🎯 Fine-tuning

### Training Frameworks
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) - Hugging Face training library
- [OpenAI Cookbook - LoRA Fine-tuning](https://cookbook.openai.com/examples/gpt-oss-lora-finetune) - Official LoRA example
- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning framework
- [QLoRA](https://github.com/artidoro/qlora) - Quantized fine-tuning

### Hardware Requirements
- **gpt-oss-120b**: Single H100 node for LoRA fine-tuning
- **gpt-oss-20b**: Consumer hardware compatible
- **Techniques**: LoRA, QLoRA, Parameter-Efficient Fine-Tuning (PEFT)

## 📱 Applications

### Chatbots and Assistants
- [Anything LLM](https://github.com/Mintplex-Labs/anything-llm) - Private document chatbot
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI-powered search engine
- [Dify](https://github.com/langgenius/dify) - LLM application development platform
- [FlowiseAI](https://github.com/FlowiseAI/Flowise) - Visual LLM app builder

### Coding Assistants
- [Aider](https://github.com/paul-gauthier/aider) - AI pair programming
- [GPT Engineer](https://github.com/gpt-engineer-org/gpt-engineer) - Code generation from specs
- [Open Interpreter](https://github.com/KillianLucas/open-interpreter) - Local code interpreter
- [MetaGPT](https://github.com/geekan/MetaGPT) - Multi-agent software development

### Research and Analysis
- [Paper QA](https://github.com/whitead/paper-qa) - Scientific paper analysis
- [LlamaIndex](https://github.com/run-llama/llama_index) - Document indexing and search
- [RAG Flow](https://github.com/infiniflow/ragflow) - Retrieval-Augmented Generation
- [Chroma](https://github.com/chroma-core/chroma) - Vector database for AI

## 📚 Tutorials

### Getting Started
- [OpenAI Cookbook - GPT-OSS Guide](https://cookbook.openai.com/articles/gpt-oss/) - Official comprehensive guide
- [How to Run GPT-OSS Locally](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama) - Step-by-step local setup
- [GPT-OSS with vLLM](https://cookbook.openai.com/articles/gpt-oss/run-vllm) - Production deployment guide
- [Harmony Response Format](https://cookbook.openai.com/articles/gpt-oss/harmony-response-format) - Understanding the new format

### Advanced Usage
- [Fine-tuning GPT-OSS](https://cookbook.openai.com/examples/gpt-oss-lora-finetune) - Custom model training
- [Building AI Agents](https://cookbook.openai.com/articles/gpt-oss/agents-sdk) - Agent development with GPT-OSS
- [Tool Use Examples](https://github.com/openai/gpt-oss/tree/main/examples) - Browser and Python tools

### Third-party Tutorials
- [GPT-OSS Setup on AWS](https://itsmybengaluru.com/how-to-install-and-use-gpt-oss-open-source-setup-tutorial-and-aws-deployment/) - Complete AWS deployment guide
- [GPU Optimization Guide](https://huggingface.co/blog/welcome-openai-gpt-oss) - Hardware-specific optimizations
- [Docker Deployment](https://northflank.com/blog/self-host-openai-gpt-oss-120b-open-source-chatgpt) - Containerized deployment

## 🔬 Research

### Academic Papers
- [GPT-OSS Model Paper](https://openai.com/index/gpt-oss-model-card/) - Technical specifications and benchmarks
- [Mixture of Experts Research](https://arxiv.org/abs/2101.03961) - MoE architecture foundations
- [MXFP4 Quantization](https://arxiv.org/abs/2310.10537) - 4-bit quantization techniques

### Benchmarks and Evaluations
- **Reasoning**: Near-parity with o4-mini on core benchmarks
- **Coding**: Strong performance on Codeforces competitions
- **Mathematics**: Excellent results on AIME 2024 & 2025
- **Tool Use**: Superior performance on TauBench agentic evaluation
- **Health**: Outperforms proprietary models on HealthBench

### Performance Analysis
- [Simon Willison's Analysis](https://simonwillison.net/2025/Aug/5/gpt-oss/) - Independent technical review
- [Comparative Benchmarks](https://huggingface.co/blog/welcome-openai-gpt-oss) - Performance vs other models
- [Enterprise Adoption Study](https://cobusgreyling.medium.com/openai-gpt-oss-8e5d1b755e79) - Market analysis

## 🛡️ Safety

### Security Features
- [Preparedness Framework Testing](https://openai.com/index/gpt-oss-model-card/#safety) - Adversarial fine-tuning results
- [Red Teaming Challenge](https://openai.com/red-teaming-challenge/) - $500,000 safety challenge
- [Safety Advisory Group Review](https://openai.com/index/gpt-oss-model-card/#external-red-teaming) - External expert evaluation

### Safety Tools
- [Content Filtering](https://github.com/openai/moderation-api) - Content moderation tools
- [Chain-of-Thought Monitoring](https://openai.com/index/introducing-gpt-oss/#chain-of-thought) - Reasoning transparency
- [Usage Policy](https://openai.com/open-models/) - Model usage guidelines

## 👥 Community

### Discussion Forums
- [OpenAI Developer Forum](https://community.openai.com/) - Official community
- [Hugging Face Forums](https://discuss.huggingface.co/) - ML community discussions
- [Reddit r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) - Local model enthusiasts
- [Discord Servers](https://discord.gg/openai) - Real-time community chat

### GitHub Organizations
- [OpenAI](https://github.com/openai) - Official repositories
- [Hugging Face](https://github.com/huggingface) - ML ecosystem
- [vLLM Team](https://github.com/vllm-project) - Inference optimization
- [Ollama](https://github.com/ollama) - Local deployment tools

### News and Updates
- [OpenAI Blog](https://openai.com/blog/) - Official announcements
- [Hugging Face Blog](https://huggingface.co/blog) - Technical deep-dives
- [AI Research Twitter](https://twitter.com/search?q=%23gptoss) - Latest developments
- [Papers with Code](https://paperswithcode.com/method/gpt-oss) - Research tracking

## 📊 Comparison with Other Models

| Feature | GPT-OSS-120b | GPT-OSS-20b | Meta Llama 3.3 70b | DeepSeek-R1 |
|---------|--------------|-------------|-------------------|-------------|
| License | Apache 2.0 | Apache 2.0 | Custom License | MIT |
| Parameters | 117B (5.1B active) | 21B (3.6B active) | 70B | 671B (37B active) |
| Memory | 80GB | 16GB | 140GB | 340GB |
| Reasoning | ✅ High | ✅ Medium | ❌ Limited | ✅ Excellent |
| Tool Use | ✅ Native | ✅ Native | ⚠️ Basic | ✅ Advanced |
| CoT Access | ✅ Full | ✅ Full | ❌ Hidden | ✅ Full |

## 🎉 Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

### How to Contribute
1. Fork this repository
2. Create a new branch for your addition
3. Add your resource with a brief description
4. Ensure it follows the existing format
5. Submit a pull request

### Criteria for Inclusion
- Must be related to GPT-OSS models
- Should be actively maintained
- Must be publicly available
- Should provide clear value to the community

## 📄 License

This awesome list is licensed under the [CC0 1.0 Universal](LICENSE) license.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=milisp/awesome-gpt-oss&type=Date)](https://star-history.com/#milisp/awesome-gpt-oss&Date)

---

Made with ❤️ by the community. If you find this list helpful, please ⭐ star it and share with others!

> **Note**: GPT-OSS models require the harmony response format to function correctly. Always use the provided chat templates or the OpenAI harmony library for proper interaction.
