# =====================================
# Interpolation for PRP-Graph
# =====================================

dataset="covid"
# "covid robust04 touche scifact signal news dbpedia nfc fiqa hotpotqa nq"
# "dl19 dl20"

model_name="flan-t5-large"
setting_c="standard"
# "standard" "no_nearest_selection" "random_initial" "inverse_initial"
setting_a="larger"
# "standard" "larger" "sub" "swiss"
rounds="10 20 40"

# -------------------------------------
# Interpolation for "w/o ranking aggregation"
# -------------------------------------

# for i in ${rounds};
# do
# prp_file=./rerank_results/${model_name}/swiss/${dataset}/${setting_c}/round${i}.swiss;
# python interpolate.py --dataset ${dataset} --prp_file $prp_file --save_file ./rerank_results/${model_name}/swiss/${dataset}/${setting_c}/round${i}.ensemble
# done

# -------------------------------------
# Interpolation for different aggregations
# -------------------------------------

if [ "$setting_a" = "standard" ]; then
    _setting_a=""
else
    _setting_a="_$setting_a"
fi

for i in ${rounds};
do
prp_file=./rerank_results/${model_name}/pagerank/${dataset}/${setting_c}/round${i}.pagerank${_setting_a}
python interpolate.py --dataset ${dataset} --prp_file $prp_file --save_file ./rerank_results/${model_name}/pagerank/${dataset}/${setting_c}/round${i}.ensemble${_setting_a}
done

# =====================================
# Interpolation for baselines
# =====================================

# dataset=covid
# method=heapsort
# model_name=flan-t5-large

# prp_file=./rerank_results/${model_name}/${method}/run.pairwise.${dataset}.txt
# python interpolate.py --dataset ${dataset} --prp_file $prp_file --save_file ./rerank_results/${model_name}/pagerank/${dataset}/${setting_c}/round${i}.ensemble${setting_a}
