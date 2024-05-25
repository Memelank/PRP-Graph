declare -A TOPICS
TOPICS=(
    [dl19]="dl19-passage"
    [dl20]="dl20-passage"
    [covid]="beir-v1.0.0-trec-covid-test"
    [touche]="beir-v1.0.0-webis-touche2020-test"
    [news]="beir-v1.0.0-trec-news-test"
    [scifact]="beir-v1.0.0-scifact-test"
    [fiqa]="beir-v1.0.0-fiqa-test"
    [hotpotqa]="beir-v1.0.0-hotpotqa-test"
    [nq]="beir-v1.0.0-nq-test"
    [nfc]="beir-v1.0.0-nfcorpus-test"
    [dbpedia]="beir-v1.0.0-dbpedia-entity-test"
    [robust04]="beir-v1.0.0-robust04-test"
    [signal]="beir-v1.0.0-signal1m-test"
)

dataset="covid"
# "covid robust04 touche scifact signal news dbpedia nfc fiqa hotpotqa nq"
# "dl19 dl20"

echo "Running graph construction on "$dataset" ..."

model="google/flan-t5-large"
model_name="flan-t5-large"
setting_c="standard"
# "standard" "inverse_initial" "no_nearest_selection" "random_initial"
gpu=0
round=40
rounds="10 20 40"

# CUDA_VISIBLE_DEVICES=${gpu} python rankers/graph_construct.py \
#     --dataset ${dataset} \
#     --model ${model} \
#     --round_num ${round} \
#     --setting_c ${setting_c}

for i in ${rounds};
do
result=./rerank_results/${model_name}/swiss/${dataset}/${setting_c}/round${i}.swiss;
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${TOPICS[${dataset}]} ${result}
done

echo "Running graph aggregation on "$dataset" ..."
setting_a="standard"
# "standard" "larger" "sub" "swiss"

if [ "$setting_a" = "standard" ]; then
    _setting_a=""
else
    _setting_a="_$setting_a"
fi

for i in ${rounds};
do
python rankers/graph_aggregate.py \
    --dataset ${dataset} \
    --round ${i} \
    --model_name ${model_name} \
    --setting_c ${setting_c} \
    --setting_a ${setting_a}
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ${TOPICS[${dataset}]} \
  ./rerank_results/${model_name}/pagerank/${dataset}/${setting_c}/round${i}.pagerank${_setting_a};
done


