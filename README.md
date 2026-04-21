## Workshop Setup Guide (FiftyOne + Model Zoo)

This workshop uses FiftyOne with PyTorch-based vision models and remote zoo model integrations.

### Python Version

Use **Python 3.9–3.12** (recommended by FiftyOne).

- Official install docs: [FiftyOne Installation Guide](https://docs.voxel51.com/installation/index.html)

---

## Install by OS

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y python3-venv python3-dev build-essential git-all libgl1-mesa-dev
sudo apt install -y libcurl4 openssl
```

> Optional (video workflows):  
```bash
sudo apt-get install -y ffmpeg
```

Create environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel build
pip install -r requirements.txt
```

> On Ubuntu 22.04+, Debian, or RHEL/CentOS, see FiftyOne FAQ note about:
> `pip install fiftyone-db==0.4.3 fiftyone`  
> [FAQ reference](https://docs.voxel51.com/faq/index.html#what-operating-systems-does-fiftyone-support)

---

### macOS

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
brew install protobuf
```

> Optional:
```bash
brew install ffmpeg
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel build
pip install -r requirements.txt
```

---

### Windows

1. Install Python 64-bit (3.9–3.12) from [python.org](https://www.python.org/downloads/) (not Microsoft Store)
2. Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
3. Install [Git](https://git-scm.com/download/win)

Create environment and install (Command Prompt):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel build
pip install -r requirements.txt
```

Optional FFmpeg: [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)

---

## Project Requirements

From `requirements.txt`:

- `fiftyone`
- `torch`
- `torchvision`
- `transformers`
- `timm`
- `open-clip-torch`
- `umap-learn`
- `ultralytics`

Install with:

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```python
import fiftyone as fo
import fiftyone.zoo as foz
print("FiftyOne OK")
```

No errors means your install is good.

---

## Model Zoo References Used in Notebook

The notebook demonstrates both built-in and remote-source zoo model workflows.

Useful references:

- FiftyOne Model Zoo Overview: [docs](https://docs.voxel51.com/model_zoo/overview.html)
- Remote-source zoo models: [docs](https://docs.voxel51.com/model_zoo/remote.html#working-with-remotely-sourced-models)
- Remote model repo (embeddings): [harpreetsahota204/qwen3vl_embeddings](https://github.com/harpreetsahota204/qwen3vl_embeddings)
- Remote model repo (vision-language ops): [harpreetsahota204/qwen3_5_vl](https://github.com/harpreetsahota204/qwen3_5_vl)

---

## Register Remote Zoo Model Sources

```python
import fiftyone.zoo as foz

# Register GitHub-hosted remote zoo sources
foz.register_zoo_model_source("github.com/harpreetsahota204/qwen3vl_embeddings")
foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen3_5_vl")
```

You can list available models after registration:

```python
print(foz.list_zoo_models())
```

---

## Download and Load Zoo Models

### Option A: Load directly (auto-download on first load)

```python
import fiftyone.zoo as foz

clip_model = foz.load_zoo_model("clip-vit-base32-torch")
qwen_embed_model = foz.load_zoo_model("Qwen/Qwen3-VL-Embedding-2B")
qwen35_model = foz.load_zoo_model("Qwen/Qwen3.5-VL-3B-Instruct")
```

### Option B: Explicitly download first

```python
import fiftyone.zoo as foz

foz.download_zoo_model("clip-vit-base32-torch")
foz.download_zoo_model("Qwen/Qwen3-VL-Embedding-2B")
foz.download_zoo_model("Qwen/Qwen3.5-VL-3B-Instruct")
```

Then load:

```python
clip_model = foz.load_zoo_model("clip-vit-base32-torch")
```

---

## Notes

- First model load/download can take time depending on model size and network.

- If using GPU, make sure your PyTorch install matches your CUDA setup.

- Ultralytics (`ultralytics`) is included for YOLO-based training/inference workflows used in the notebook.

