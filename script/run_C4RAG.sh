pwd

python3 "model/C4RAG.py" \
    --yaml_filepath "configs/en_config.yaml" \
    --dataset "popqa" \
    --output_directory "result" \
    --device "cuda" \
    --retriever_name "hybrid"

