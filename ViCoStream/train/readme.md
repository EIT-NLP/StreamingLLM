# Training

### Download dataset

Download training data from [TimeChat-Online-139K](https://huggingface.co/datasets/wyccccc/TimeChat-Online-139K) and [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).

Then replace the video paths in the JSONL files with your local video paths.

---

### Create Conda Environment


```bash
conda create --name vicostream python=3.10
conda activate vicostream

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r ../requirements.txt

pip install 'datasets>=3.5,<4' 'pyarrow>=15,<22'
pip install 'deepspeed>=0.15.0,<0.19.0' wandb tensorboard
```

---

### Set Up `ms-swift` and `transformers`

Choose one of the following installation options.

#### Option 1: Clone the Upstream Sources and Apply the ViCoStream Patches

Use this option when `train/ms-swift` and `train/transformers` are not already present. From the ViCoStream project root, run:

```bash
git clone --branch v3.2.0 --depth 1 \
  https://github.com/modelscope/ms-swift.git \
  train/ms-swift

git clone --branch v4.49.0 --depth 1 \
  https://github.com/huggingface/transformers.git \
  train/transformers

pip install -e train/ms-swift
pip install -e train/transformers

bash train/pooling-replace-code/notes

```
#### Option 2: Use the Bundled Patched Sources (If ms-swift and transformers have been installed.)

The source trees included in this repository already contain the ViCoStream patches. From the ViCoStream project root, run:

```bash
pip install -e train/ms-swift
pip install -e train/transformers
```

> The replacement script applies six ViCoStream-specific source overrides. It must be run from the ViCoStream project root.

---

### Launch Training Script

Edit these placeholders in [`finetune.sh`](./finetune.sh):

```bash
MODEL_PATH="Path/to/your/model"
OUTPUT_DIR="Path/to/your/output/dir"
--dataset "Path/to/your/dataset-1" "Path/to/your/dataset-2"
```

Then run:

```bash
cd /Path/to/ViCoStream
conda activate vicostream
bash train/finetune.sh
```

The script uses `chunk-intra` dropping, `chunk_size=4`, `attend_chunk_num=4`, `user_query_retrieval=16`, and `MAX_PIXELS=90000` by default.

---

### Notes

- The six patched files are kept in [`pooling-replace-code/`](./pooling-replace-code/). They are already applied to `train/ms-swift` and `train/transformers`.
- `datasets==5.x` can break ms-swift's Arrow writer patch. Use `datasets>=3.5,<4`.
- Always use `PYTHONNOUSERSITE=1` and `pip install --no-user` if your machine has packages in `~/.local`.
- The default `finetune.sh` uses `--attn_impl eager`, so `flash-attn` is not required. If you switch to flash attention, install a compatible `flash-attn` package first.
