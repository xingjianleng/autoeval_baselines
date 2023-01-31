methods=(atc confscore entropy fid rotation)
models=(resnet repvgg)
GPU_ID=0

for i in "${models[@]}"
do
    for j in "${methods[@]}"
    do
        echo -e "\nRunning $j method with model: $i"
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 baselines/"$j.py" "$i"
    done
done
