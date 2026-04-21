## Workshop Setup Guide (FiftyOne + Model Zoo)

This workshop uses FiftyOne with PyTorch-based vision models and remote zoo model integrations.

- Python: **3.9–3.12** (recommended by FiftyOne)
- Official install docs: [FiftyOne Installation Guide](https://docs.voxel51.com/installation/index.html)

---

## 1. Install System Prerequisites

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install -y python3-venv python3-dev build-essential git-all libgl1-mesa-dev libcurl4 openssl
# Optional (video workflows):
sudo apt-get install -y ffmpeg
```

> On Ubuntu 22.04+, Debian, or RHEL/CentOS, see the FiftyOne FAQ note about pinning `fiftyone-db`:
> `pip install fiftyone-db==0.4.3 fiftyone` — [FAQ reference](https://docs.voxel51.com/faq/index.html#what-operating-systems-does-fiftyone-support).

### macOS

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 protobuf
# Optional:
brew install ffmpeg
```

### Windows

1. Install Python 64-bit (3.9–3.12) from [python.org](https://www.python.org/downloads/) (not Microsoft Store)
2. Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
3. Install [Git](https://git-scm.com/download/win)
4. Optional FFmpeg: [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)

---

## 2. Create Environment & Install Python Packages

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel build
pip install -r requirements.txt
```

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel build
pip install -r requirements.txt
```

`requirements.txt` pulls in: `fiftyone`, `torch`, `torchvision`, `transformers`, `timm`, `open-clip-torch`, `umap-learn`, `ultralytics`.

---

## 3. Verify Installation & Launch the App

Run the quickstart — it both verifies the install and opens the FiftyOne App:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
session.wait()  # keep the app session alive when running as a script
```

One-liner equivalent:

```bash
python -c "import fiftyone as fo, fiftyone.zoo as foz; d=foz.load_zoo_dataset('quickstart'); s=fo.launch_app(d); s.wait()"
```

If the App opens and shows samples, your install is fully working.

---

## 4. Port Forwarding for Remote Sessions

If FiftyOne is running on a remote machine, forward the App port to your local machine.

**On the remote machine:**

```python
import fiftyone as fo

dataset = fo.load_dataset("<dataset-name>")
session = fo.launch_app(dataset, remote=True)  # optional: port=XXXX
```

**On your local machine** (recommended — requires FiftyOne installed locally):

```bash
fiftyone app connect --destination [<username>@]<hostname>
# Custom ports / SSH key:
fiftyone app connect --destination [<username>@]<hostname> --port XXXX --local-port YYYY --ssh-key /path/to/key
```

**Manual SSH forwarding** (fallback), then open `http://localhost:5151`:

```bash
ssh -N -L 5151:127.0.0.1:XXXX [<username>@]<hostname>
```

**Multiple remote sessions** — start each on a different port and forward each to a different local port:

```python
session1 = fo.launch_app(dataset1, remote=True, port=XXXX)
session2 = fo.launch_app(dataset2, remote=True, port=YYYY)
```

```bash
ssh -N -L WWWW:localhost:XXXX [<username>@]<hostname>
ssh -N -L ZZZZ:localhost:YYYY [<username>@]<hostname>
```

**Docker** — expose the App port:

```bash
docker run -p 5151:5151 ...
```

References:
- [Remote sessions](https://docs.voxel51.com/user_guide/app.html#remote-sessions)
- [Connect to remote App](https://docs.voxel51.com/cli/index.html#connect-to-remote-app)
- [Remote data](https://docs.voxel51.com/installation/environments.html#remote-data)
- [Serve multiple remote sessions](https://docs.voxel51.com/faq/index.html#can-i-serve-multiple-remote-sessions-from-a-machine) · [Connect to multiple](https://docs.voxel51.com/faq/index.html#can-i-connect-to-multiple-remote-sessions)
- [Docker environment](https://docs.voxel51.com/installation/environments.html#docker)

---

## 5. Model Zoo Setup

The notebook uses both built-in and remote-source zoo models.

**Register remote zoo sources, then load models:**

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen3vl_embeddings")
foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen3_5_vl")

# List everything that's now available
print(foz.list_zoo_models())

# Load models (auto-downloads on first load; use foz.download_zoo_model(name) to pre-fetch)
clip_model       = foz.load_zoo_model("clip-vit-base32-torch")
qwen_embed_model = foz.load_zoo_model("Qwen/Qwen3-VL-Embedding-2B", media_type="image")
qwen35_model     = foz.load_zoo_model("Qwen/Qwen3.5-2B", media_type="image", operation="classify")
```

References:
- [FiftyOne Model Zoo](https://docs.voxel51.com/model_zoo/overview.html) · [Remote-source models](https://docs.voxel51.com/model_zoo/remote.html#working-with-remotely-sourced-models)
- Remote repos: [qwen3vl_embeddings](https://github.com/harpreetsahota204/qwen3vl_embeddings) · [qwen3_5_vl](https://github.com/harpreetsahota204/qwen3_5_vl)

---

## Notes

- First model load/download can take time depending on model size and network.
- If using GPU, make sure your PyTorch install matches your CUDA setup.
- `ultralytics` is included for the YOLO-based workflows used in the notebook.
