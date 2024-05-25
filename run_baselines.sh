declare -A NAME
NAME=(
    [covid]="trec-covid"
    [touche]="webis-touche2020"
    [news]="trec-news"
    [scifact]="scifact"
    [fiqa]="fiqa"
    [hotpotqa]="hotpotqa"
    [nq]="nq"
    [nfc]="nfcorpus"
    [dbpedia]="dbpedia-entity"
    [robust04]="robust04"
    [signal]="signal1m"
)

dataset="covid"
# "covid robust04 touche scifact signal news dbpedia nfc fiqa hotpotqa nq"
# "dl19 dl20"
method="heapsort"
# "heapsort" "bubblesort" "allpair"
model="google/flan-t5-large"
model_name="flan-t5-large"
# shuffle_ranking="random"
gpu=0

echo "Running "${method}" on "${dataset}" ..."
CUDA_VISIBLE_DEVICES=${gpu} python3 run_baselines.py \
  run --model_name_or_path ${model} \
      --tokenizer_name_or_path ${model} \
      --run_path ./retrieve_results/BM25/trec_results_${dataset}.txt \
      --save_path ./rerank_results/${model_name}/${method}/run.pairwise.${dataset}.txt \
      --pyserini_index beir-v1.0.0-${NAME[${dataset}]} \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
      --device cuda \
  pairwise --method ${method} \
           --k 10;

python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 beir-v1.0.0-${NAME[${dataset}]}-test \
  ./rerank_results/${model_name}/${method}/run.pairwise.${dataset}.txt;


