for city in Beijing Porto San_Francisco; do
    python code/preprocess.py --city $city
done

python code/main.py --city Beijing --seq_len 64 --pretrain
python code/main.py --city Porto --seq_len 296 --pretrain
python code/main.py --city San_Francisco --seq_len 296 --pretrain

for city in Beijing Porto San_Francisco; do
    python code/eval.py \
        --city $city \
        --roadmap_geo_path data/${city}/roadmap.geo \
        --label_trajs_path preprocessed/${city}/test.data \
        --gen_trajs_path preprocessed/${city}/gen_test.data
done