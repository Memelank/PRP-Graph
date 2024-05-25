import networkx as nx
import numpy as np


class Pagerank():
    def __init__(
        self,
        setting_a,
        compare_records,
        weights
    ) -> None:
        self.setting_a = setting_a
        self.compare_records = compare_records
        self.weights = weights
    

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x))
        probabilities = exp_values / np.sum(exp_values, axis=0)
        return probabilities

    def run_pagerank(self):
        results = {}
        for compare_record in self.compare_records:
            qid = list(compare_record.keys())[0]
            compare_record = list(compare_record.values())[0]
            
            if "larger" == self.setting_a:
                edges = []
                done_node = set()
                for data in compare_record:
                    target_node = data["docid"]
                    score = data["compare_score_record"]
                    compare_docid = data["compare_record"]
                    for idx, source_node in enumerate(compare_docid):
                        if source_node not in done_node:
                            score_s2t = score[idx]
                            for tmp in compare_record:
                                if tmp["docid"] == source_node:
                                    score_t2s = tmp["compare_score_record"][tmp["compare_record"].index(target_node)]
                                    if score_s2t > score_t2s:
                                        edges.append((source_node, target_node, {"weight":score_s2t}))
                                    else:
                                        edges.append((target_node, source_node, {"weight":score_t2s}))
                    done_node.add(target_node)
            elif "sub" == self.setting_a:
                edges = []
                done_node = set()
                for data in compare_record:
                    target_node = data["docid"]
                    score = data["compare_score_record"]
                    compare_docid = data["compare_record"]
                    for idx, source_node in enumerate(compare_docid):
                        if source_node not in done_node:
                            score_s2t = score[idx]
                            for tmp in compare_record:
                                if tmp["docid"] == source_node:
                                    score_t2s = tmp["compare_score_record"][tmp["compare_record"].index(target_node)]
                                    if score_s2t > score_t2s:
                                        edges.append((source_node, target_node, {"weight":score_s2t-score_t2s}))
                                    else:
                                        edges.append((target_node, source_node, {"weight":score_t2s-score_s2t}))
                    done_node.add(target_node)
            else:
                edges = []
                for data in compare_record:
                    target_node = data["docid"]
                    score = data["compare_score_record"]
                    compare_docid = data["compare_record"]
                    for idx, source_node in enumerate(compare_docid):
                        edges.append((source_node, target_node, {"weight":score[idx]}))
                    
            G = nx.DiGraph()    
            G.add_edges_from(edges)
            
            weight = self.weights[qid]
            G.add_nodes_from([list(x.keys())[0] for x in weight])


            initial_weights = {}
            for data in weight:
                initial_weights.update(data)


            pagerank = nx.pagerank(G, personalization=initial_weights, weight='weight')

            pagerank = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
            results[qid] = pagerank
        return results