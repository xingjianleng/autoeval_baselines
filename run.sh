methods=(atc confscore entropy fid rotation)
models=(resnet repvgg)

for i in "${models[@]}"
do
    echo -e "\nCalculate accuracy for model: $i"
    python3 "accuracy.py" "$i"
done

for i in "${models[@]}"
do
    for j in "${methods[@]}"
    do
        echo -e "\nRunning $j method with model: $i"
        python3 "$j.py" "$i"
    done
done
