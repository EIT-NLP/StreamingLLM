<h1 align="center"><b>LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding</b></h1>
</div>


<p align="center">
<a href="https://huggingface.co/spaces/JunlongTong/" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/arxiv-2405.xxxxx-DA644E?logo=arxiv" /></a>
<a href="https://arxiv.org/" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/🤗 Hugging Face Models-2980b9?color=2980b9" /></a>

</p>





The code will be released soon.


## Batch-processing vs. streaming-processing
### batch-processing: The LLMs process the entire input at once.
![batch-processing](./asset/batch.gif)
### streaming-processing: The LLMs process the input as it arrives, incrementally and in real time.
#### Interleaved-streaming:
![streaming-processing](./asset/interleaved.gif)
####  Group-streaming:
![batch-processing](./asset/streaming.gif)


<!-- ## Introduction
Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability.
This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: 
* **Input-attention mismatch**:
* **Output-attention mismatch**:
* **Position-ID mismatch mismatch**:
 -->
