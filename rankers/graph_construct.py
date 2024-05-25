import json
from argparse import ArgumentParser
import os
import random

from swiss_system import swiss_system
from rank_flan import FlanT5
random.seed(42)


def load_data(path):
    with open(path) as file:
        origin_datas = json.load(file)
    datas = []
    for d in origin_datas:
        data = {}
        data["query"] = d["query"]
        if not len(d["hits"]):
            continue
        data["qid"] = d["hits"][0]["qid"]
        hits = d["hits"]
        retrieval_docid = []
        retrieval_doc = {}
        for h in hits:
            docid = h["docid"]
            retrieval_docid.append(docid)
            retrieval_doc[docid] = h["content"]
        data["retrieval_docid"] = retrieval_docid
        data["retrieval_doc"] = retrieval_doc
        datas.append(data)
    return datas

def write_result(rerank_datas, dataset, round, model_name, setting_c):
    output_root = f"./rerank_results/{model_name}/swiss/{dataset}/{setting_c}"
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_file = os.path.join(output_root, f"round{round}.swiss")
    with open(output_file, 'w') as f:
        for qid in rerank_datas["ranking"].keys():
            for i, pid in enumerate(rerank_datas["ranking"][qid]):
                f.write(f'{qid} Q0 {pid} {i+1} {rerank_datas["score"][qid][i]} rank\n')

    return output_file

if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", default="google/flan-t5-large", type=str)
    parser.add_argument("--round_num", default=40, type=int)
    parser.add_argument("--setting_c", default="standard", type=str)

    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    round_num = args.round_num
    setting_c = args.setting_c

    model_name = model.split('/')[-1]
    data_file = f"./retrieve_results/BM25/retrieve_results_{dataset}.json"
    datas = load_data(data_file)
    
    agent = FlanT5(
            model=model,
        )
    generate_func = agent.generate

    round_records, results = swiss_system(datas, generate_func, setting_c, round_num)

    for r in range(1, round_num+1):
        records = round_records[f"round {r}"]
        pagerank_root = f"./graph_data/{model_name}/{dataset}/{setting_c}"
        os.makedirs(pagerank_root, exist_ok=True)
        with open(os.path.join(pagerank_root, f"round{r}"), "w") as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

    for i in [10,20,40]:
        output_file = write_result(results[f"round {i}"], dataset, i, model_name, setting_c)