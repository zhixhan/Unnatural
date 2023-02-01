for i in {1..500}
do
    export AKEY=$(az account get-access-token --resource https://ml.azure.com -o tsv | cut -d $'\t' -f1)
    python input_generation.py --seed_task "12" --batch_dir "data/gpt3_generations_12/"
done