# CPU  pipeline setup

How to install the speech-recognition models the pipeline needs. This mainly
covers **downloading and placing Vosk models** (the CPU path).

---

## Languages the pipeline supports

The pipeline has **two ASR backends**, chosen by `device:` in
`config/config.yaml` (`auto` = GPU if available, else CPU). **Which languages
you can transcribe depends entirely on which backend is active.**

| Backend | When it's used | Language coverage |
| --- | --- | --- |
| **Whisper** (`asr.gpu`, faster-whisper) | `device: gpu`, or `auto` when a CUDA GPU is present | ~99 languages. This is the pipeline's intended target. |
| **Vosk** (`asr.cpu`) | `device: cpu`, or `auto` with no GPU | Only languages that have a Vosk model  ~30 languages (see table below). |

>  **Important Hungarian language caveat.** There is **no Hungarian Vosk model**. On the
> CPU path the pipeline cannot transcribe Hungarian it will load whatever
> model `asr.cpu.model_path`. If you need real
> Hungarian transcription, use the **GPU/Whisper path**.
9

---

## How Vosk models work 

- Vosk models are **directories**, not single files. You download a `.zip`,
  extract it, and point the config at the extracted folder.
- The folder is referenced by **`asr.cpu.model_path`** in `config/config.yaml`.
  It's currently:

  ```yaml
  asr:
    cpu:
      engine: "vosk"
      model_path: "models/vosk-model-small-en-us"
  ```

- **This path is relative to where you launch the pipeline.** `main.py` is meant
  to be run from `src/`, so the default resolves to **`src/models/vosk-model-small-en-us`**.
- Two model tiers exist:
  - **small** (~40–80 MB): real-time on a CPU core → **use these for this pipeline.**
  - **big** (0.3–2+ GB): higher accuracy, needs far more RAM/CPU → will likely
    fall behind live audio. Avoid unless you've measured it keeps up.

---

## Install steps

### 1. Install the Vosk Python package

```bash
pip install vosk
```

### 2. Download a model

Pick a model from the [tables below](#available-vosk-models). Every model lives at:

```
https://alphacephei.com/vosk/models/<MODEL-NAME>.zip
```

Download it (browser, or command line):

```bash
# from the src/ directory
mkdir -p models
cd models

# example: small US-English model
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
# or: wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
```

### 3. Extract it

```bash
unzip vosk-model-small-en-us-0.15.zip
# -> creates the folder: vosk-model-small-en-us-0.15/
```

### 4. Make the config match the folder name

The extracted folder is `vosk-model-small-en-us-0.15`, but the config expects
`vosk-model-small-en-us`. Pick **one** of:

**Option A: rename the folder** (keeps the config untouched):

```bash
mv vosk-model-small-en-us-0.15 vosk-model-small-en-us
```

**Option B: update the config** to the real folder name:

```yaml
asr:
  cpu:
    model_path: "models/vosk-model-small-en-us-0.15"
```

Either way, the final layout should be:

```
src/
└── models/
    └── vosk-model-small-en-us/     <-- asr.cpu.model_path points here
        ├── am/
        ├── conf/
        ├── graph/
        └── ...
```

### 5. Verify

Force the CPU path and run transcribe-only (no sound clips needed):

```bash
# in config.yaml set:  device: cpu
python main.py --transcribe-only
```

You should see the Vosk model load and live `~`/`=` transcript lines appear. If
the model path is wrong you'll get a clear "Vosk model directory not found"
error naming the path it tried.

---

## Available Vosk models

Small models are the real-time-friendly choice for this pipeline. Big/server
models are listed for reference only.

### Recommended (small, real-time)

| Language | Model name (append `.zip` to the base URL) | Size |
| --- | --- | --- |
| English (US) | `vosk-model-small-en-us-0.15` | 40 MB |
| Indian English | `vosk-model-small-en-in-0.4` | 36 MB |
| Chinese | `vosk-model-small-cn-0.22` | 42 MB |
| Russian | `vosk-model-small-ru-0.22` | 45 MB |
| French | `vosk-model-small-fr-0.22` | 41 MB |
| German | `vosk-model-small-de-0.15` | 45 MB |
| Spanish | `vosk-model-small-es-0.42` | 39 MB |
| Portuguese | `vosk-model-small-pt-0.3` | 31 MB |
| Turkish | `vosk-model-small-tr-0.3` | 35 MB |
| Vietnamese | `vosk-model-small-vn-0.4` | 32 MB |
| Italian | `vosk-model-small-it-0.22` | 48 MB |
| Dutch | `vosk-model-small-nl-0.22` | 39 MB |
| Catalan | `vosk-model-small-ca-0.4` | 42 MB |
| Farsi/Persian | `vosk-model-small-fa-0.42` | 53 MB |
| Ukrainian | `vosk-model-small-uk-v3-small` | 133 MB |
| Kazakh | `vosk-model-small-kz-0.42` | 58 MB |
| Swedish | `vosk-model-small-sv-rhasspy-0.15` | 289 MB |
| Japanese | `vosk-model-small-ja-0.22` | 48 MB |
| Esperanto | `vosk-model-small-eo-0.42` | 42 MB |
| Hindi | `vosk-model-small-hi-0.22` | 42 MB |
| Czech | `vosk-model-small-cs-0.4-rhasspy` | 44 MB |
| Polish | `vosk-model-small-pl-0.22` | 50 MB |
| Uzbek | `vosk-model-small-uz-0.22` | 49 MB |
| Korean | `vosk-model-small-ko-0.22` | 82 MB |
| Breton | `vosk-model-br-0.8` | 70 MB |
| Gujarati | `vosk-model-small-gu-0.42` | 100 MB |
| Tajik | `vosk-model-small-tg-0.22` | 50 MB |
| Telugu | `vosk-model-small-te-0.42` | 58 MB |
| Kyrgyz | `vosk-model-small-ky-0.42` | 49 MB |
| Georgian | `vosk-model-small-ka-0.42` | 45 MB |
| Arabic (Tunisian) | `vosk-model-small-ar-tn-0.1-linto` | 158 MB |
| Filipino (Tagalog) | `vosk-model-tl-ph-generic-0.6` | 320 MB |
| Greek | `vosk-model-el-gr-0.7` | 1.1 GB (no small model) |
| Arabic | `vosk-model-ar-mgb2-0.4` | 318 MB (no small model) |

### Big / server models (accuracy over latency — not recommended here)

| Language | Model name | Size |
| --- | --- | --- |
| English (US) | `vosk-model-en-us-0.22` | 1.8 GB |
| English (Gigaspeech) | `vosk-model-en-us-0.42-gigaspeech` | 2.3 GB |
| Chinese | `vosk-model-cn-0.22` | 1.3 GB |
| Russian | `vosk-model-ru-0.42` | 1.8 GB |
| French | `vosk-model-fr-0.22` | 1.4 GB |
| German | `vosk-model-de-0.21` | 1.9 GB |
| Spanish | `vosk-model-es-0.42` | 1.4 GB |
| Italian | `vosk-model-it-0.22` | 1.2 GB |
| Japanese | `vosk-model-ja-0.22` | 1 GB |
| Hindi | `vosk-model-hi-0.22` | 1.5 GB |
| Kazakh | `vosk-model-kz-0.42` | 1.3 GB |
| Kyrgyz | `vosk-model-ky-0.42` | 1.1 GB |
| Georgian | `vosk-model-ka-0.42` | 700 MB |
| Gujarati | `vosk-model-gu-0.42` | 700 MB |
| Tajik | `vosk-model-tg-0.22` | 327 MB |
| Farsi/Persian | `vosk-model-fa-0.42` | 1.6 GB |
| Ukrainian | `vosk-model-uk-v3` | 343 MB |
| Vietnamese | `vosk-model-vn-0.4` | 78 MB |
| Arabic | `vosk-model-ar-0.22-linto-1.1.0` | 1.3 GB |

> The complete, authoritative, up-to-date list is at
> <https://alphacephei.com/vosk/models>. New languages are added over time.

---

## If you need Hungarian

Vosk has no Hungarian model, so:

1. **Use the GPU / Whisper path (recommended).** Set `device: gpu` (or `auto`
   with a CUDA GPU present). Whisper supports Hungarian and `asr.gpu.language`
   is already `"hu"`. No model download needed beyond faster-whisper pulling the
   Whisper weights on first run.
2. **Find a community Hungarian Kaldi/Vosk model.** None is official; quality
   varies. It must follow the Vosk model directory layout to load.
3. **Train your own** with the Vosk/Kaldi recipe (needs a CUDA GPU to train and
   a Hungarian speech corpus, e.g. Common Voice Hungarian). Significant effort.

