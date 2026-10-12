# AI Device Capability Inspector — Neural Telemetry & WebAI Benchmark

[![GitHub Pages](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen)](https://sureshmagix.github.io/ai-device-inspector/)
[![WebGPU](https://img.shields.io/badge/Accelerator-WebGPU%20%7C%20WGSL%20FP16-06b6d4)](https://www.w3.org/TR/webgpu/)
[![Wasm SIMD](https://img.shields.io/badge/Runtime-Wasm%20SIMD%20%7C%20WebNN-6366f1)](https://webassembly.org/)
[![Portfolio](https://img.shields.io/badge/Portfolio-sureshmagix.github.io-08756f)](https://sureshmagix.github.io/)

A client-side hardware diagnostics and neural compute benchmark suite that inspects your browser's AI acceleration stack (**WebGPU**, **FP16 shaders**, **VRAM buffer limits**, **Wasm 128-bit SIMD**, **WebNN**, and **IndexedDB/Cache storage headroom**) to determine readiness for in-browser LLMs (`WebLLM`), embeddings (`Transformers.js`), and computer vision (`ONNX Runtime Web`).

👉 **Live Interactive Demo**: **[https://sureshmagix.github.io/ai-device-inspector/](https://sureshmagix.github.io/ai-device-inspector/)**  
👈 **More Projects & Experiments**: **[https://sureshmagix.github.io/](https://sureshmagix.github.io/)**

---

## 🌟 Key Features

### 1. Real-Time Micro-Benchmarks (WGSL Compute & CPU Tensor Math)
* **WebGPU WGSL Compute Benchmark**: Compiles and dispatches 24 iterations of a custom `@workgroup_size(64)` WGSL compute shader to measure real GPU floating-point throughput (`GFLOPS`).
* **CPU FP32 Tensor Dot-Product Benchmark**: Runs an unrolled `Float32Array` fused multiply-add (FMA) benchmark to quantify single-core JS/CPU tensor throughput.
* **Model Storage Quota Audit**: Queries `navigator.storage.estimate()` to verify whether the browser has enough disk headroom to cache multi-gigabyte 4-bit quantized LLM weights (`q4f16_1`).

### 2. Deep WebAI & Accelerator Telemetry
* **Modern WebGPU Inspection**: Reads synchronous `adapter.info` (with fallback to `requestAdapterInfo()` and unmasked WebGL2 `WEBGL_debug_renderer_info`) alongside `shader-f16` support, `maxBufferSize`, and `maxStorageBufferBindingSize`.
* **Wasm SIMD & WebNN Probes**: Validates a 128-bit SIMD WebAssembly module in-memory and checks for `navigator.ml` (WebNN) and Chrome Built-in AI (`window.ai` / `LanguageModel`).
* **5-Axis SVG Capability Radar & Readiness Score**: Visualizes **WebGPU**, **VRAM**, **CPU**, **Storage**, and **AI APIs** on an interactive pentagon radar chart and calculates a `0–100` readiness score across 4 hardware tiers (`Tier S` to `Tier C`).

### 3. Browser AI Model Compatibility Matrix
Evaluates real-world compatibility and estimated inference speeds for:
* **Embeddings / RAG**: `all-MiniLM-L6-v2` (~85 MB, Transformers.js SIMD)
* **Speech & Vision**: `Whisper-Base / YOLOv8n` (~145 MB, ONNX Runtime Web)
* **Compact SLMs (1B–3B)**: `Llama-3.2-1B / Gemma-2-2B` (~1.3 GB `q4f16_1`, WebLLM)
* **Full Edge LLMs (7B–8B)**: `Qwen-2.5-7B / Llama-3.1-8B` (~4.6 GB `q4f16_1`, WebLLM)

### 4. AI Training Planner & Epoch Runtime Estimates
Plan a realistic fine-tuning run before opening a local or cloud notebook:
* **Curated task-specific model choices**: DistilBERT / ModernBERT for text, BGE / E5 for retrieval, MobileNetV3 / ViT for vision, Whisper Tiny / Base for speech, and Qwen / Llama for LLM adapter tuning.
* **Interactive estimate inputs**: Configure dataset size, epochs, batch size, and token length, image size, or audio duration. The tool returns steps per epoch plus per-epoch and end-to-end wall-clock ranges.
* **Measured-device calibration**: When WebGPU is available, the range is scaled from the page's shader benchmark; otherwise it intentionally uses a wider CPU baseline range.
* **Training-fit guidance**: Each model shows its recommended fine-tuning method, approximate accelerator-memory range, download size, and appropriate use case.

The planner is for **capacity planning**, not browser training. WebGPU buffer limits are browser allocations and do not report installed GPU VRAM. For dependable results, run 50–100 optimizer steps in your actual PyTorch/JAX/other training environment and multiply the median step time by the displayed step count. Add time for tokenization, data loading, evaluation, checkpointing, and model downloads.

---

## 🚀 Running Locally

Because **WebGPU** (`navigator.gpu`) and **Storage Quota** (`navigator.storage.estimate()`) require a **Secure Context** (`http://localhost` or `https://`), serve the directory over localhost:

```bash
python3 -m http.server 8080
```

Then open **[http://localhost:8080](http://localhost:8080)** in Chrome, Edge, or Brave.
