> **AI-powered personalized children's movies** — çdo fëmijë bëhet hero i filmit të tij.

Dërgo foton, emrin dhe datëlindjen e fëmijës → FalconAI gjeneron një film të personalizuar ku fëmija është heroi kryesor.

---

```
foto.jpg + "Erind" + "2018-05-10"
            ↓
    [Face Processor]      nxjerr fytyrën me InsightFace
            ↓
    [Story Generator]     gjeneron skenarin me Mistral 7B
            ↓
    [Frame Generator]     gjeneron imazhet me Stable Diffusion + IP-Adapter
            ↓
    [Audio Generator]     gjeneron tregimin me TTS + muzikë sfond
            ↓
    [Video Assembler]     bashkon gjithçka → MP4 finale
            ↓
    data/outputs/erind_musliu_20240115.mp4  ✅
```

---

```
falconai-kids/
│
├── main.py
├── config/
│   └── settings.py
│
├── pipeline/
│   ├── orchestrator.py
│   ├── face_processor.py
│   ├── story_generator.py
│   ├── frame_generator.py
│   ├── audio_generator.py
│   └── video_assembler.py
│
├── utils/
│   ├── logger.py
│   ├── validators.py
│   └── exceptions.py
│
├── scripts/
│   ├── download_models.py
│   └── test_gpu.py
│
├── data/
│   ├── inputs/
│   ├── outputs/
│   └── temp/
│
├── models_cache/
├── requirements.txt
├── .env.example
└── README.md
```

---


| Komponent | Minimum | Rekomanduar |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB    | 24 GB (NVIDIA A10G) |
| RAM       | 16 GB   | 32 GB |
| Disk      | 25 GB   | 50 GB |
| Python    | 3.10    | 3.11 |
| CUDA      | 11.8    | 12.1 |

**AWS Instance:** `g5.xlarge` (NVIDIA A10G, 24GB VRAM) — rekomandohet.

---

```bash
git clone https://github.com/yourusername/falconai-kids.git
cd falconai-kids
```

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

```bash
sudo apt update && sudo apt install ffmpeg

brew install ffmpeg

```

```bash
cp .env.example .env
```

Ndrysho `.env` sipas sistemit tënd:

```env
DEVICE=cuda
LOG_LEVEL=INFO
HUGGINGFACE_TOKEN=hf_your_token_here # nga huggingface.co/settings/tokens
```

> Shko te [huggingface.co/mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) dhe kliko **"Agree and access repository"**.

```bash
python scripts/test_gpu.py
```

```bash
python scripts/download_models.py
```

---

```bash
python main.py \
  --photo data/inputs/foto.jpg \
  --name "Erind Musliu" \
  --birthday 2018-05-10
```

```bash
python main.py \
  --photo foto.jpg \
  --name "Erind" \
  --birthday 2018-05-10 \
  --language English

python main.py \
  --photo foto.jpg \
  --name "Erind" \
  --birthday 2018-05-10 \
  --no-audio

python main.py \
  --photo foto.jpg \
  --name "Erind" \
  --birthday 2018-05-10 \
  --no-audio --no-upscale

python main.py \
  --photo foto.jpg \
  --name "Erind" \
  --birthday 2018-05-10 \
  --verbose
```

| Opsion | Default | Përshkrim |
|--------|---------|-----------|
| `--photo` | — | Rruga e fotos (e detyrueshme) |
| `--name` | — | Emri i fëmijës (i detyrueshëm) |
| `--birthday` | — | Datëlindja YYYY-MM-DD (e detyrueshme) |
| `--language` | Albanian | Albanian, English, Italian, German, French |
| `--output-dir` | data/outputs | Folder i videos finale |
| `--no-audio` | False | Gjenero pa audio |
| `--no-upscale` | False | Kalo upscaling |
| `--no-cleanup` | False | Mos fshi temp files |
| `--seed` | random | Seed për riprodhueshmëri |
| `--verbose` | False | Log i detajuar |

---

| Model | Madhësia | Roli |
|-------|----------|------|
| InsightFace buffalo_l | ~500MB | Face detection + embedding |
| Mistral 7B Instruct v0.2 | ~14GB | Gjenerim historie |
| Stable Diffusion v1.5 | ~4GB | Gjenerim imazhesh |
| IP-Adapter SD1.5 | ~300MB | Integrimi i fytyrës |
| AnimateDiff v1.5 | ~1.8GB | Animim |
| Coqui TTS | ~800MB | Text-to-Speech |
| RealESRGAN x4 | ~300MB | Upscaling |

**Total:** ~22GB hapësirë disk

---

```bash
python scripts/download_models.py

python scripts/download_models.py --model face
python scripts/download_models.py --model llm
python scripts/download_models.py --model sd
python scripts/download_models.py --model anim
python scripts/download_models.py --model tts
python scripts/download_models.py --model esrgan

python scripts/download_models.py --check
```

---

```bash
python main.py --photo foto.jpg --name "Erind" --birthday 2018-05-10 --no-upscale

```

```bash
```

```bash
echo "HUGGINGFACE_TOKEN=hf_xxx" >> .env
```

```bash
sudo apt install ffmpeg

ffmpeg -version
```

```bash
pip install TTS

```

---

Bazuar në **AWS G5.xlarge (NVIDIA A10G, 24GB VRAM)**:

| Hapi | Koha |
|------|------|
| Face Processing | ~5 sekonda |
| Story Generation | ~30 sekonda |
| Frame Generation (5 skena × 16 frames) | ~8-12 minuta |
| Audio Generation | ~1 minuta |
| Video Assembly | ~30 sekonda |
| Upscaling | ~2 minuta |
| **Total** | **~12-16 minuta** |

---

| Gjuha | Kodi | TTS |
|-------|------|-----|
| Shqip | Albanian | tts_models/sq/cv/vits |
| Anglisht | English | tts_models/en/ljspeech/tacotron2-DDC |
| Italisht | Italian | tts_models/it/mai_female/vits |
| Gjermanisht | German | tts_models/de/thorsten/vits |
| Frëngjisht | French | tts_models/fr/mai/vits |

---

- **[InsightFace](https://github.com/deepinsight/insightface)** — Face detection & recognition
- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)** — Image generation
- **[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)** — Identity-preserving generation
- **[AnimateDiff](https://github.com/guoyww/AnimateDiff)** — Video animation
- **[Mistral 7B](https://mistral.ai/)** — Story generation
- **[Coqui TTS](https://github.com/coqui-ai/TTS)** — Text-to-speech
- **[RealESRGAN](https://github.com/xinntao/Real-ESRGAN)** — Video upscaling
- **[FFmpeg](https://ffmpeg.org/)** — Video processing
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** — ML pipelines

---

MIT License — shiko [LICENSE](LICENSE) për detaje.

---