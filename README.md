# PRP-Graph

This repository contains the code and resources for our paper:

> Jian Luo, Xuanang Chen, Ben He, Le Sun. PRP-Graph: Pairwise Ranking Prompting to LLMs with Graph Aggregation for Effective Text Re-ranking. ACL 2024.

------figure 1

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
The PRP-Graph operates in two main stages: ranking graph construction and ranking graph aggregation. 

### Construction
For ranking graph construction, document pairs are selectively compared according to Swiss-System to form a ranking graph with documents as vertices linked by bidirectional edges.
Following is an example of run graph construction and get the re-ranking results of this stage:
```
CUDA_VISIBLE_DEVICES=0 python rankers/graph_construct.py \
    --dataset covid \
    --model google/flan-t5-large \
    --round_num 40 

result=./rerank_results/flan-t5-large/swiss/covid/standard/round40.swiss;
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 beir-v1.0.0-trec-covid-test ${result}

Results:
ndcg_cut_10             all     0.7701
```

### Aggregation
For ranking graph aggregation, the signals across the ranking graph constructed by the first stage are aggregated to produce a cohesive final document ranking that encapsulates the entire graph's sorting information.
The following is an example of how to aggregate the ranking graph and obtain re-ranking results with PRP-Graph:
```
python rankers/graph_aggregate.py \
    --dataset covid \
    --round 40 \
    --model_name flan-t5-large

result=./rerank_results/flan-t5-large/pagerank/covid/standard/round40.pagerank
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 beir-v1.0.0-trec-covid-test ${result}

Results:
ndcg_cut_10             all     0.7880
```

## Interporation with BM25

## Cite
