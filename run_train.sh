for city in Beijing Porto San_Francisco; do
    python code/preprocess.py --city $city
done

for city in Porto San_Francisco; do
    python code/main.py --city $city --seq_len 296 --pretrain
done

python code/main.py --city Beijing --seq_len 64 --pretrain