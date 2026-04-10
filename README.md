# WebAssembly in Node.js Performance Benchmark

Dieses Projekt vergleicht die Performance von: 
- JavaScript,
- WebAssembly (Rust) und
- WebAssembly (Rust) mit lane-wise SIMD

jeweils mit und ohne Einsatz von Worker-Threads im Node.js-Backend über den Bildverarbeitungsalgorithmus "Gaussian Blur".

## Struktur:

**gaussian_blur_wasm / gaussian_blur_wasm_simd**:
  - enthält die Rust-Implementierung sowie das jeweilige kompilierte WASM-Modul des Gaussian Blur.

**/src**: 
  - /bench: vollständiger Compute-Benchmark
  - /variants: Implementierungsvarianten und Worker Pool
  - gaussian_blur.js: JavaScript Implementierung des Gaussian Blur Algorithmus
  - server.js: Server für den Systembenchmark

**/tools**: 
  - Beispielbilder
  - Äquivalenztests
  - Load-Generator für den Systembenchmark
  - TestImageGenerator

## Ausführung
Der Compute-Benchmark wird über runner.js gestartet, der System-Benchmark nach starten des Servers über 
loadgen.js.