import copy
import json
import random
import re
import os
from tqdm import tqdm
random.seed(42)

global compare_count
compare_count = 0

def PRP_unit(query, passage_a, passage_b, generate, ovonic=True):
    if ovonic:
        score = generate(query, passage_a, passage_b)    
    global compare_count
    compare_count += 1
    return score

def swiss_system(datas, generate, setting_c, round_num=7):
    class Player:
        def __init__(self, docid, text, weight) -> None:
            self.docid = docid
            self.text = text
            self.compare_record = []
            self.compare_score_record = []
            self.score = weight
            self.round = 0

        def add_score(self, score, player):
            self.compare_score_record.append(score[0])
            player.compare_score_record.append(score[1])

            tmp = self.score
            self.score += score[0]*player.score/player.round
            player.score += score[1]*tmp/self.round


        def add_compare_record(self, compare_docid):
            self.compare_record.append(compare_docid)
            self.round += 1


    round_results = {f"round {x}":{"ranking":{}, "score":{}} for x in range(1, round_num+1)}
    round_records = {f"round {x}":[] for x in range(1, round_num+1)}
    for data in tqdm(datas):
        rank_list = data["retrieval_docid"]
        corpus = data["retrieval_doc"]
        query = data["query"]

        if len(rank_list) == 1:
            result = {}
            result[data["qid"]] = [{
            "docid":rank_list[0],
            "compare_record":[],
            "compare_score_record":[]
            }]

            for i in range(1, round_num+1):
                round_records[f"round {i}"].append(result)
                round_results[f"round {i}"]["ranking"][data["qid"]] = rank_list
                round_results[f"round {i}"]["score"][data["qid"]] = [1]

            continue
        
        if setting_c == "no_nearest_selection" or setting_c == "random_initial":
            random.shuffle(rank_list)
        if setting_c == "inverse_initial":
            rank_list.reverse()

        bye = None
        
        # print(rank_list)
        if setting_c == "no_nearest_selection":
            players = [Player(docid=doc, text=corpus[doc], weight=1) for i, doc in enumerate(rank_list)]        
        else:
            players = [Player(docid=doc, text=corpus[doc], weight=1-i*0.01) for i, doc in enumerate(rank_list)]        
        
 
        if len(players)%2 != 0:
            bye = players[-1]
            players = players[:len(players)-1]

        group_1 = [players[idx] for idx in [x for x in range(len(players)) if x%2==0]]
        group_2 = [players[idx] for idx in [x for x in range(len(players)) if x%2!=0]]
        
        for i in range(len(group_1)):
            group_1[i].add_compare_record(group_2[i].docid)
            group_2[i].add_compare_record(group_1[i].docid)
            passage_a=group_1[i].text
            passage_b=group_2[i].text
            score = PRP_unit(query, passage_a, passage_b, generate)
            group_1[i].add_score(score, group_2[i])

        players = sorted(players, key = lambda x:x.score, reverse=True)
        docid_result = [x.docid for x in players]
        if bye:
            docid_result.append(bye.docid)
        score_result = [x.score for x in players]
        if bye:
            score_result.append(bye.score)
        round_results["round 1"]["ranking"][data["qid"]] = docid_result
        round_results["round 1"]["score"][data["qid"]] = score_result

        for r in range(2, round_num+1):
            # select group
            group_1 = []
            group_2 = []
            for i, player_1 in enumerate(players):
                if player_1.round == r:
                    continue
                for j, player_2 in enumerate(players[i+1:]):
                    if player_2.docid in player_1.compare_record or player_2.round == r:
                        continue
                    group_1.append(player_1)
                    group_2.append(player_2)
                    player_1.add_compare_record(player_2.docid)
                    player_2.add_compare_record(player_1.docid)
                    break
            
            for i in range(len(group_1)):
                passage_a=group_1[i].text
                passage_b=group_2[i].text
                score = PRP_unit(query, passage_a, passage_b, generate)
                group_1[i].add_score(score, group_2[i])

            if setting_c == "no_nearest_selection":
                tmp = copy.deepcopy(players)

            players = sorted(players, key = lambda x:x.score, reverse=True)

            docid_result = [x.docid for x in players]
            if bye:
                docid_result.append(bye.docid)
                
            score_result = [x.score for x in players]
            if bye:
                score_result.append(bye.score)
            round_results[f"round {r}"]["ranking"][data["qid"]] = docid_result
            round_results[f"round {r}"]["score"][data["qid"]] = score_result

            result = {}
            result[data["qid"]] = [{
                "docid":x.docid,
                "compare_record":copy.deepcopy(x.compare_record),
                "compare_score_record":copy.deepcopy(x.compare_score_record)
                } for x in players]
            
            if bye:
                result[data["qid"]].append({
                    "docid":bye.docid,
                    "compare_record":[],
                    "compare_score_record":[]
                    })
            round_records[f"round {r}"].append(result)

            if setting_c == "no_nearest_selection":
                players = tmp
        # print("round_records:")
        # print(round_records)
        # print("round_results:")
        # print(round_results)
        # exit()
    
    
    global compare_count

    print("ave compare num:", compare_count/len(datas))

    return round_records, round_results
