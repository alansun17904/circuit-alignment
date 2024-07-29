# export HF_HOME=/BRAIN/circuit-alignment/work/cache

# python3 baseline.py gpt2-xl gpt2 "gpt2-xl-norm-single.pkl" --batch_size 64 --text_type norm --avg_subjs
# python3 baseline.py gpt2-xl gpt2 "gpt2-xl-hf-plato-single.pkl" --batch_size 64 --text_type plato --avg_subjs
# python3 baseline.py gpt2-xl gpt2 "gpt2-xl-randtok-single.pkl" --batch_size 64 --text_type randtok --avg_subjs
# python3 baseline.py gpt2-xl gpt2 "gpt2-xl-poa-single.pkl" --batch_size 64 --text_type poa --avg_subjs


# python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-norm-single.pkl" --batch_size 64 --text_type norm --hf_model --hf_model_id meta-llama/Llama-2-7b-hf --avg_subjs
# python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-plato-single.pkl" --batch_size 64 --text_type plato --hf_model --hf_model_id meta-llama/Llama-2-7b-hf --avg_subjs
# python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-randtok-single.pkl" --batch_size 64 --text_type randtok --hf_model --hf_model_id meta-llama/Llama-2-7b-hf --avg_subjs
# python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-poa-single.pkl" --batch_size 64 --text_type poa --hf_model --hf_model_id meta-llama/Llama-2-7b-hf --avg_subjs

# python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-norm-single.pkl" --batch_size 32 --text_type norm --hf_model --hf_model_id meta-llama/Llama-2-13b-hf --avg_subjs
# python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-plato-single.pkl" --batch_size 32 --text_type plato --hf_model --hf_model_id meta-llama/Llama-2-13b-hf --avg_subjs
# python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-randtok-single.pkl" --batch_size 32 --text_type randtok --hf_model --hf_model_id meta-llama/Llama-2-13b-hf --avg_subjs
# python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-poa-single.pkl" --batch_size 32 --text_type poa --hf_model --hf_model_id meta-llama/Llama-2-13b-hf --avg_subjs

# python3 evaluate.py gpt2-small gpt2 all data/bbench-baseline/gpt2-small-3.json --shots 3
python3 evaluate.py gpt2-medium gpt2 all data/bbench-baseline/gpt2-medium-3.json --shots 3
python3 evaluate.py gpt2-large gpt2 all data/bbench-baseline/gpt2-large-3.json --shots 3
python3 evaluate.py gpt2-xl gpt2 all data/bbench-baseline/gpt2-xl-3.json --shots 3