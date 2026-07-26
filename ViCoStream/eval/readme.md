# Evaluation

## Requirements

Use the same `vicostream` environment as training.

```bash
conda create -n vicostream python=3.10
conda activate vicostream

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r ../requirements.txt

# Explicitly install decord. Otherwise qwen-vl-utils may fall back to torchvision/PyAV.
pip install decord==0.6.0
pip install 'openai>=1.0.0' ffmpeg-python==0.2.0 moviepy==1.0.3
```

For StreamBench and VStream-QA, configure the GPT judge:

```bash
export VSTREAM_EVAL_API_KEY="sk-..."
export VSTREAM_EVAL_API_BASE="your_base_url"
export VSTREAM_EVAL_MODEL="gpt-3.5-turbo"
```

---

## Benchmarks

All incremental launchers are in [`scripts/eval_inc/`](./scripts/eval_inc/). Set `CKPT_PATH`, dataset paths, and `RESULT_DIR` in the corresponding shell before running.


### StreamingBench

Download the dataset from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench). Only `StreamingBench/Real_Time_Visual_Understanding.csv` and `Real-Time Visual Understanding_*.zip` are needed.

In [`eval_incremental_streamingbench.sh`](./scripts/eval_inc/eval_incremental_streamingbench.sh), set `TASK_CSV` and `VIDEO_DIR`. Then run:

```bash
bash eval/scripts/eval_inc/eval_incremental_streamingbench.sh
```

### OVO-Bench

Download videos from [JoeLeelyf/OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench), and download `ovo_bench_new.json` from the OVO-Bench GitHub repo.

In [`eval_incremental_ovobench.sh`](./scripts/eval_inc/eval_incremental_ovobench.sh), set `TASK_JSON` and `VIDEO_DIR`. Then run:

```bash
bash eval/scripts/eval_inc/eval_incremental_ovobench.sh
```

### ETBench

Prepare [ETBench](https://huggingface.co/datasets/PolyU-ChenLab/ETBench/tree/main), then set `DATA_ROOT`, `SOURCE_JSON`, `TASK_JSON`, and `METRIC_SCRIPT` in [`eval_incremental_etbench.sh`](./scripts/eval_inc/eval_incremental_etbench.sh).

```bash
bash eval/scripts/eval_inc/eval_incremental_etbench.sh
```

### StreamBench v0.3

Set `DATA_ROOT`, `TASK_JSON`, and `SOURCE_JSON` in [`eval_incremental_streambench.sh`](./scripts/eval_inc/eval_incremental_streambench.sh).

```bash
bash eval/scripts/eval_inc/eval_incremental_streambench.sh
```

### VStream-QA

Set `TASK_JSON` and `VIDEO_DIR` in the corresponding script:

```bash
bash eval/scripts/eval_inc/eval_incremental_vstream_ego4d.sh
bash eval/scripts/eval_inc/eval_incremental_vstream_movienet.sh
```

---

## Arguments

In each `eval_incremental_*.sh`, there are several alterable arguments:

- `RUN_NAME`: The name of this run, used for logging.
- `CKPT_PATH`: The path to the checkpoint for evaluation.
- `RESULT_DIR`: The directory used to store logs, outputs and predictions.
- `DROP_METHOD`: The drop method. The default is `chunk-intra`.
- `DROP_THRESHOLD`: The drop ratio or threshold.
- `CHUNK_SIZE`: 1/2 Number of frames in each streaming chunk during token dropping.
- `ATTEND_CHUNK_NUM`: Number of previous chunks visible to attention.
- `USER_QUERY_RETRIEVAL`: Number of previous visual chunks retrieved for user queries.
- `MAX_PIXELS` / `VIDEO_MAX_PIXELS`: Frame resolution budget.

---

## Notes

- `eval/scripts/evaluate_open_ended_qa.py` is required by StreamBench and VStream-QA.
- If `qwen-vl-utils` prints `using torchvision to read video`, install `decord==0.6.0` in the active environment.
