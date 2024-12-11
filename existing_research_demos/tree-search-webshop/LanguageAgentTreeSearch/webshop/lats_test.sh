export OPENAI_API_KEY=sk-proj-8PcGgZqkhEJeW62qrHz0T3BlbkFJdgCgMebKWG8pU3TM2O24

python run.py \
    --backend gpt-4 \
    --task_start_index 0 \
    --task_end_index 10 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 5 \
    --log logs/new_run_$(date +"%Y%m%d_%H%M%S").log \
    ${@}

# remember to change the url in lats.py to your local instance of WebShop 

