# AI Device Capability Inspector — Neural Telemetry & WebAI Benchmark

[![GitHub Pages](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen)](https://sureshmagix.github.io/ai-device-inspector/)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA%20Jetson-Orin%20%7C%20Xavier%20%7C%20Nano-76b900)](https://developer.nvidia.com/embedded-computing)
[![WebGPU](https://img.shields.io/badge/Accelerator-WebGPU%20%7C%20WGSL%20FP16-06b6d4)](https://www.w3.org/TR/webgpu/)
[![Wasm SIMD](https://img.shields.io/badge/Runtime-Wasm%20SIMD%20%7C%20WebNN-6366f1)](https://webassembly.org/)
[![Portfolio](https://img.shields.io/badge/Portfolio-sureshmagix.github.io-08756f)](https://sureshmagix.github.io/)

A client-side hardware diagnostics, neural compute benchmark suite, and edge AI inspector designed for desktop PCs, laptops, and **NVIDIA Jetson edge boards (Orin AGX/NX/Nano, Xavier, Nano, TX2)**. It probes your device's acceleration stack (**WebGPU**, **WebGL 2.0 GPGPU**, **FP16 half-float shaders**, **Unified Memory Architecture [UMA]**, **Wasm 128-bit SIMD**, **WebNN**, and **local storage quota**) to assess real-world readiness for in-browser LLMs (`WebLLM`), embeddings (`Transformers.js`), and edge vision models (`ONNX Runtime Web`).

👉 **Live Interactive Demo**: **[https://sureshmagix.github.io/ai-device-inspector/](https://sureshmagix.github.io/ai-device-inspector/)**  
👈 **More Projects & Experiments**: **[https://sureshmagix.github.io/](https://sureshmagix.github.io/)**

---

## 🌟 Key Features

### 1. First-Class NVIDIA Jetson & Edge Board Detection
* **Jetson Architecture Recognition**: Automatically detects and classifies NVIDIA Tegra SoC series:
  * **Jetson Orin Series** (Tegra234, Ampere architecture, 512–2048 CUDA cores, 3rd Gen Tensor Cores, DLA 2.0, 20–275 TOPS INT8).
  * **Jetson Xavier Series** (Tegra194, Volta architecture, 384–512 CUDA cores, 1st Gen Tensor Cores, DLA 1.0, 21–32 TOPS INT8).
  * **Jetson Nano** (Tegra210, Maxwell architecture, 128 CUDA cores, 472 GFLOPS FP16).
  * **Jetson TX2** (Tegra186, Pascal architecture, 256 CUDA cores, 1.3 TFLOPS FP16).
* **Unified Memory (UMA) Recognition**: Recognizes that Jetson boards share system LPDDR4/LPDDR5 memory directly between CPU and CUDA GPU (up to 64GB on AGX Orin), removing discrete VRAM limitations.
* **JetPack / L4T Driver Audit**: Identifies the installed JetPack / Linux for Tegra driver generation (JetPack 6.x / 5.x / 4.x) directly from OpenGL ES subsystem strings.
* **WebGPU Activation Helper**: Provides a 1-click terminal command to launch Chromium on Jetson Linux with hardware-accelerated Vulkan and WebGPU flags enabled:
  ```bash
  chromium-browser --enable-features=Vulkan,UseSkiaRenderer,DefaultANGLEVulkan --enable-unsafe-webgpu
  ```

### 2. Dual-Engine Live GPU Micro-Benchmarks
* **WebGPU WGSL Compute Benchmark**: When WebGPU is supported, dispatches 24 passes of an intensive `@workgroup_size(64)` WGSL compute shader measuring direct GPU floating-point throughput (`GFLOPS`).
* **WebGL2 GPGPU Shader Fallback**: When WebGPU is dormant or disabled (standard default in Chromium on Linux / Jetson), compiles and runs an unrolled 512×512 fragment shader FMA benchmark (~5.03 GFLOPs per run) to benchmark the NVIDIA Tegra GPU directly via WebGL2!
* **CPU FP32 Tensor Dot-Product Benchmark**: Runs an unrolled `Float32Array` fused multiply-add (FMA) benchmark to quantify single-core CPU throughput.

### 3. Deep System, Graphics & Platform Telemetry
* **Processor & Compute Pipeline**: Architecture (`aarch64` / `arm64` / `x86_64`), logical core count, Wasm 128-bit SIMD, Wasm Threads & SharedArrayBuffer (`crossOriginIsolated`), and WebNN (`navigator.ml`).
* **Graphics & Shader Subsystem**: WebGL version (2.0 vs 1.0), unmasked GPU renderer, Max Texture Size (e.g. 16384×16384 px on Orin), Max Renderbuffer Size, FP16 Half-Float Texture support (`EXT_color_buffer_half_float`), and 32-bit float texture rendering.
* **Memory & Storage Subsystem**: Reported RAM cap, Unified Memory Architecture (UMA) vs discrete VRAM, total quota, and available cache headroom for model weights (`navigator.storage.estimate()`).
* **Display & Visual Engine**: Screen resolution, Device Pixel Ratio (DPR), color depth, and live measured refresh rate / FPS (`requestAnimationFrame`).
* **Audio DSP Hardware**: AudioContext sample rate (`48000 Hz` / `44100 Hz`), base latency, and channel count (essential for Whisper Speech AI).
* **Connectivity & Power Topology**: Connection type, downlink bandwidth, RTT, and AC mains vs battery power status (`navigator.getBattery()`).

### 4. Browser AI Model Compatibility Matrix
Evaluates real-world compatibility and estimated inference speeds for:
* **Embeddings / RAG**: `all-MiniLM-L6-v2` (~85 MB, Transformers.js SIMD)
* **Speech & Vision**: `Whisper-Base / YOLOv8n` (~145 MB, ONNX Runtime Web)
* **Compact SLMs (1B–3B)**: `Llama-3.2-1B / Gemma-2-2B` (~1.3 GB `q4f16_1`, WebLLM / Jetson UMA)
* **Full Edge LLMs (7B–8B)**: `Qwen-2.5-7B / Llama-3.1-8B` (~4.6 GB `q4f16_1`, WebLLM / Jetson UMA)

### 5. AI Training Planner **Version 2.0**
Plan a realistic fine-tuning run before opening a local or cloud notebook:
* **17 curated profiles across five workloads**: Text (DistilBERT, DeBERTa-v3, ModernBERT), retrieval (BGE-small, BGE-base, E5-base), vision (MobileNetV3, EfficientNet-B0, ViT), speech (Whisper Tiny/Base/Small), and LLMs (Qwen, Gemma, Llama, Mistral).
* **Run-aware inputs**: Configure dataset size, validation split, epochs, micro-batch, gradient accumulation, device count, sequence/image/audio size, tuning strategy, precision, data-pipeline efficiency, evaluation/checkpoint cadence, and hourly accelerator rates.
* **Measured-device calibration**: Scaled from the page's live measured GPU / CPU throughput benchmarks.

---

## 🚀 Running Locally

Because **WebGPU** (`navigator.gpu`) and **Storage Quota** (`navigator.storage.estimate()`) require a **Secure Context** (`http://localhost` or `https://`), serve the directory over localhost:

```bash
python3 -m http.server 8080
```

Then open **[http://localhost:8080](http://localhost:8080)** in Chrome, Edge, or Brave.

On **NVIDIA Jetson boards**, launch Chromium with WebGPU flags enabled:
```bash
chromium-browser --enable-features=Vulkan,UseSkiaRenderer,DefaultANGLEVulkan --enable-unsafe-webgpu
```
