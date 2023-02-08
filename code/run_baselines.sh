methods=(atc confscore entropy fid rotation)
models=(resnet repvgg)
GPU_ID=0

dataset_path=/data/lengx/cifar/

for i in "${models[@]}"
do
    echo -e "\nCalculating dataset accuracy for model: $i"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 get_accuracy.py --model "$i" --dataset_path "$dataset_path"
done

for i in "${models[@]}"
do
    for j in "${methods[@]}"
    do
        echo -e "\nRunning $j method with model: $i"
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 baselines/"$j.py" --model "$i" --dataset_path "$dataset_path"
    done
done
