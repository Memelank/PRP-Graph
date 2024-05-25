import random

random.seed(42)

def read_result(result_file):
    results = {}
    with open(result_file, 'r') as file:
        for line in file:
            [qid, _, docid, _, score, _] = line.split()

            if qid not in results:
                results[qid] = []

            results[qid].append([docid, float(score)])
    return results

def getKey(x):
    return x[1]

def write_result(result_file, result, need_sort=False):
    with open(result_file, 'w') as f:
        for query in result:
            score_list = []
            for item in result[query]:
                score_list.append(item)
            if need_sort:
                score_list.sort(key=getKey, reverse=True)
            for index, item in enumerate(score_list):
                f.write(f'{query}\tQ0\t{item[0]}\t{index}\t{item[1]}\tEnsemble\n')

def get_ensemble_result(result_1, result_2, alpha=0.5):
    ensemble = {}
    for query in result_1:
        ensemble[query] = []
        for doc_1, score_1 in result_1[query]:
            for doc_2, score_2 in result_2[query]:
                if doc_1 == doc_2:
                    ensemble[query].append([doc_1, score_1*alpha + score_2*(1-alpha)])
    return ensemble


def normalizer(result):
    for query in result:
        max = 0
        min = 1000000
        if len(result[query]) == 1:
            continue
        for doc, score in result[query]:
            if score > max:
                max = score
            if score < min:
                min = score
        max_minus_min = max - min
        for i in range(len(result[query])):
            result[query][i][1] = (result[query][i][1] - min) / max_minus_min

def ndcg_api(dataset, result_file):
    import subprocess
    from prp_graph.retriever.topics_dict import TOPICS
    args = ['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], result_file]

    output = subprocess.check_output(['python', '-m', "pyserini.eval.trec_eval"] + args, stderr=subprocess.STDOUT, universal_newlines=True)
    # print(output)
    for line in output.split("\n"):
        if "all" in line and "allpair" not in line:
            ndcg  = float(line.split()[-1])
    return ndcg

def split_dict(dictionary):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    part = len(keys) // 10
    dict1 = {key: dictionary[key] for key in keys[:part]}
    dict2 = {key: dictionary[key] for key in keys[part:part*2]}
    dict3 = {key: dictionary[key] for key in keys[part*2:part*3]}
    dict4 = {key: dictionary[key] for key in keys[part*3:part*4]}
    dict5 = {key: dictionary[key] for key in keys[part*4:part*5]}
    dict6 = {key: dictionary[key] for key in keys[part*5:part*6]}
    dict7 = {key: dictionary[key] for key in keys[part*6:part*7]}
    dict8 = {key: dictionary[key] for key in keys[part*7:part*8]}
    dict9 = {key: dictionary[key] for key in keys[part*8:part*9]}
    dict10 = {key: dictionary[key] for key in keys[part*9:]}
    return dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--prp_file', required=True)
    parser.add_argument('--save_file', required=True)
    # parser.add_argument('--method', required=True)
    args = parser.parse_args()


    dataset = args.dataset
    prp_file = args.prp_file
    save_file = args.save_file
    # method = args.method

    bm_file = f'./retrieve_results/BM25/trec_results_{dataset}.txt'
    bm_result = read_result(bm_file)
    prp_result = read_result(prp_file)

    normalizer(bm_result)
    normalizer(prp_result)

    prp_group_1, prp_group_2, prp_group_3, prp_group_4, prp_group_5, \
    prp_group_6, prp_group_7, prp_group_8, prp_group_9, prp_group_10  = split_dict(prp_result)
    
    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_1}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_1 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_2}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_2 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_3}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_3 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_4}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_4 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_5}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_5 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_6}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_6 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_7}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_7 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_8}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_8 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_9}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_9 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ndcg_result = []
    for alpha in [x * 0.1 for x in range(11)]:
        ensemble_result = get_ensemble_result({key: value for key, value in prp_result.items() if key not in prp_group_10}, bm_result, alpha)
        write_result('./tmp.auto_ensemble', ensemble_result, need_sort=True)
        ndcg_result.append(ndcg_api(dataset, './tmp.auto_ensemble'))
    alpha_10 = 0.1*(ndcg_result.index(max(ndcg_result)))

    ensemble_result = {}
    ensemble_result.update(get_ensemble_result(prp_group_1, bm_result, alpha_1))
    ensemble_result.update(get_ensemble_result(prp_group_2, bm_result, alpha_2))
    ensemble_result.update(get_ensemble_result(prp_group_3, bm_result, alpha_3))
    ensemble_result.update(get_ensemble_result(prp_group_4, bm_result, alpha_4))
    ensemble_result.update(get_ensemble_result(prp_group_5, bm_result, alpha_5))
    ensemble_result.update(get_ensemble_result(prp_group_6, bm_result, alpha_6))
    ensemble_result.update(get_ensemble_result(prp_group_7, bm_result, alpha_7))
    ensemble_result.update(get_ensemble_result(prp_group_8, bm_result, alpha_8))
    ensemble_result.update(get_ensemble_result(prp_group_9, bm_result, alpha_9))
    ensemble_result.update(get_ensemble_result(prp_group_10, bm_result, alpha_10))
    write_result(save_file, ensemble_result, need_sort=True)
    ndcg_result = ndcg_api(dataset, save_file)
    print("alpha_1:", alpha_1)
    print("alpha_2:", alpha_2)
    print("alpha_3:", alpha_3)
    print("alpha_4:", alpha_4)
    print("alpha_5:", alpha_5)
    print("alpha_6:", alpha_6)
    print("alpha_7:", alpha_7)
    print("alpha_8:", alpha_8)
    print("alpha_9:", alpha_9)
    print("alpha_10:", alpha_10)
    print("ndcg@10:", ndcg_result)
