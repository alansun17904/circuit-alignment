python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-plato.pkl" --batch_size 64 --text_type plato --hf_model --hf_model_id meta-llama/Llama-2-7b-hf
python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-randtok.pkl" --batch_size 64 --text_type randtok --hf_model --hf_model_id meta-llama/Llama-2-7b-hf
python3 baseline.py Llama-2-7b meta-llama/Llama-2-7b-hf "llama2-7b-hf-poa.pkl" --batch_size 64 --text_type poa --hf_model --hf_model_id meta-llama/Llama-2-7b-hf

python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-norm.pkl" --batch_size 32 --text_type norm --hf_model --hf_model_id meta-llama/Llama-2-13b-hf
python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-plato.pkl" --batch_size 32 --text_type plato --hf_model --hf_model_id meta-llama/Llama-2-13b-hf
python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-randtok.pkl" --batch_size 32 --text_type randtok --hf_model --hf_model_id meta-llama/Llama-2-13b-hf
python3 baseline.py Llama-2-13b meta-llama/Llama-2-13b-hf "llama2-13b-hf-poa.pkl" --batch_size 32 --text_type poa --hf_model --hf_model_id meta-llama/Llama-2-13b-hf