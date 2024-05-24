# PRP-Graph

This repository contains the code and resources for our paper:

> Jian Luo, Xuanang Chen, Ben He, Le Sun. PRP-Graph: Pairwise Ranking Prompting to LLMs with Graph Aggregation for Effective Text Re-ranking. ACL 2024.

## Installation
We recommend you create a new conda environment `conda create -n prp_graph python=3.9`, 
activate it `conda activate prp_graph`, and then install the following packages:
```
accelerate==0.23.0
pyserini==0.22.1
torch==2.1.0
transformers==4.33.1
faiss-cpu==1.7.4
networkx==3.1
ir_datasets
```
## First-stage retrieval
PRP-Graph uses LLMs to re-rank top documents retrieved by a first-stage retriever. We implement BM25 retriever based on [rank_llm](https://github.com/castorini/rank_llm). Run the commands below to get the first-stage runs and data.
```bash
cd retriever/
python pyserini_retriever.py
```
## Ranking Graph Construction and Aggregation

## Baselines

## Interporation with BM25

## Cite
