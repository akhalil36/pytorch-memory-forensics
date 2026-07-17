# PyTorch Model Memory Forensics

A deep-dive into runtime memory forensics of a PyTorch convolutional neural network. This project combines systems programming, machine learning, and memory forensics to extract a neural network's weights, architecture, and forward pass code directly from a raw memory image without access to the original source code or model file.

**[→ Full write-up, findings, and analysis on the GitHub Pages site](https://akhalil36.github.io/pytorch-memory-forensics/)**

---

## What This Project Does

This project **extracts a trained neural network directly from physical memory** using low-level forensics tools. Given only a memory dump of a running process, this project recovers:

- The complete model architecture (layer names, types, and order)
- Trained weights and biases for every layer
- The forward pass execution path
- Process memory layout

---

## Technical Stack

- **PyTorch 2.5.1** — Model training and TorchScript export
- **C++ / LibTorch** — Inference binary with full debug symbols
- **GDB** — Runtime inspection of PyTorch data structures
- **LiME** — Linux kernel module for raw memory capture
- **Volatility 3** — Memory forensics framework
- **dwarf2json** — DWARF debug symbol extraction
- **x86_64 Ubuntu 24.04 LTS** — Target system (emulated via UTM on Apple Silicon)

---

## Project Structure

```
pytorch-memory-forensics/
├── model/
│   ├── model.py                        # Custom CNN architecture
│   ├── train.py                        # Training script with 80/10/10 train/val/test split
│   └── export.py                       # TorchScript export for use in C++ inference
├── inference/
│   ├── CMakeLists.txt                  # Build config with full debug symbols enabled
│   └── cpp_inference/
│       └── main.cpp                    # Loads TorchScript model and runs inference
├── artifacts/
│   ├── cnn_model.pt                    # PyTorch saved model weights
│   └── cnn_model_torchscript.pt        # TorchScript model used by the C++ binary
├── custom_volatility_plugin/
│   ├── torch_model_extractor_plugin.py # Volatility 3 plugin: recovers weights, architecture, forward pass
│   └── custom_plugin_output.txt        # Output of running the custom Volatility plugin
├── docs/
│   └── index.html                      # Full project write-up with findings and screenshots
├── env.yml                             # Conda environment with PyTorch dependencies
└── setup.sh                            # Downloads LibTorch (CPU) and the CIFAR-10 dataset
```

---

## Key Technical Challenges

**ARM64 vs x86_64 Compatibility** — Volatility 3's Linux plugins are broken on ARM64 kernels 6.8+. The project required migrating from an ARM64 VM to an emulated x86_64 VM to get working memory forensics.

**Symbol Table Generation** — Matching the exact kernel build (not just version) between the memory dump and the DWARF symbol table is critical. A mismatch between architecture banners causes Volatility to silently fail.

**PyTorch Memory Layout** — PyTorch's C++ data structures (`TensorImpl`, `StorageImpl`, `IValue`, `ClassType`) are not documented for forensic use. The [PyTorch GitHub source](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h) was used to map struct field offsets manually.

---

## References

- [The Art of Memory Forensics](https://www.wiley.com/en-us/The+Art+of+Memory+Forensics-p-9781118825099)
- [PyTorch TensorImpl.h](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h)
- [LiME — Linux Memory Extractor](https://github.com/504ensicsLabs/LiME)
- [Volatility 3 Framework](https://github.com/volatilityfoundation/volatility3)
- [dwarf2json](https://github.com/volatilityfoundation/dwarf2json)