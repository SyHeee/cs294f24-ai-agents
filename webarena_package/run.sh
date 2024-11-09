#!/bin/bash
export HOSTNAME="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com"
export SHOPPING="${HOSTNAME}:7770"
export SHOPPING_ADMIN="${HOSTNAME}:7780/admin"
export REDDIT="${HOSTNAME}:9999"
export GITLAB="${HOSTNAME}:8023"
export MAP="${HOSTNAME}:3000"
export WIKIPEDIA="${HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="${HOSTNAME}:4399" # this is a placeholder
echo "Done setting up URLs"

# Make sure the above code runs successfully.
rm -rf ./results/*.html
# python run.py \
#   --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
#   --test_start_idx 0 \
#   --test_end_idx 1 \
#   --model gpt-3.5-turbo \
#   --result_dir ./results/


python minimal_example.py