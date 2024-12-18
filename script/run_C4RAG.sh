pwd

python3 "model/REFIND.py" \
    --yaml_filepath "configs/en_config.yaml" \
    --dataset "popqa" \
    --output_directory "result" \
    --device "cuda" \
    --retriever_name "hybrid"

