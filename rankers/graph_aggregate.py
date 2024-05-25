import json
from argparse import ArgumentParser
import os

from pagerank import Pagerank

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--round", type=int, default=10)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--model_name", type=str)
parser.add_argument("--setting_c", default="standard",type=str)
parser.add_argument("--setting_a", default="standard", type=str)

args = parser.parse_args()
dataset = args.dataset
round = args.round
output = args.output
model_name = args.model_name
setting_c = args.setting_c
setting_a = args.setting_a

weights = {}
if "swiss" == setting_a:
    with open(f"./rerank_results/{model_name}/swiss/{dataset}/{setting_c}/round{round}.swiss") as f:
        for line in f:
            if '\t' in line:
                q_id, _, p_id, rank, score, _ = line.strip().split('\t')
            else:
                q_id, _, p_id, rank, score, _ = line.strip().split(' ')
            if q_id not in weights:
                weights[q_id] = [{p_id:float(score)}]
            else:
                weights[q_id].append({p_id:float(score)})
else:
    with open(f"./retrieve_results/BM25/trec_results_{dataset}.txt") as f:
        for line in f:
            q_id, _, p_id, rank, score, _ = line.strip().split(' ')
            if q_id not in weights:
                weights[q_id] = [{p_id:float(score)}]
            else:
                weights[q_id].append({p_id:float(score)})

compare_records = []
with open(f"./graph_data/{model_name}/{dataset}/{setting_c}/round{round}") as f:
    lines = f.readlines()
    for line in lines:
        compare_records.append(json.loads(line))

pagerank = Pagerank(compare_records=compare_records,setting_a=setting_a,weights=weights)

results = pagerank.run_pagerank()

if setting_a == "standard":
    setting_a = ""
else:
    setting_a = '_'+setting_a



output_root = f"./rerank_results/{model_name}/pagerank/{dataset}/{setting_c}"
os.makedirs(output_root, exist_ok=True)

with open(os.path.join(output_root, f"round{round}.pagerank{setting_a}"), 'w') as f:
    for qid, pagerank in results.items():
        i = 1
        for pid,score in pagerank.items():
            f.write(f'{qid} Q0 {pid} {i} {score} pagerank\n')
            i+=1
