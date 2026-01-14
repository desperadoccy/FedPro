#!/bin/bash
set -e  # Exit strictly on error

# ================= Configuration =================
# GPU Setting (Modify here or use env var: GPU_ID=0,1 ./run_experiments.sh)
: "${GPU_ID:=0}"

# ================= Commands List =================
# EDIT THIS ARRAY to add or modify experiments
# Format: "Complete Command String"
declare -a CMDS=(

    "python loop_df_fl.py --type=kd_train --iid=0 --epochs=80 --local_ep=1000 --lr=0.005 --batch_size 16 --synthesis_batch_size=8 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/siri-ours-accumuT-nosyn --model=res --dataset=siri-whu --adv=1 --beta=0.5 --seed=3407 --alg=ours --num_users=10 --checkpoint_steps 100 200 300 500 800 1300 2000 --weights_dir=/data/ccy/fedproto/20260112_104826/weights --smooth_transition --output_dir=/data/ccy/fedproto/ --continuous_teacher --no_wandb"

    "python loop_df_fl.py --type=kd_train --iid=0 --epochs=80 --local_ep=1000 --lr=0.005 --batch_size 32 --synthesis_batch_size=8 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/siri-ours-accumuT-nosyn --model=res --dataset=siri-whu --adv=1 --beta=0.5 --seed=3407 --alg=ours --num_users=10 --checkpoint_steps 100 200 300 500 800 1300 2000 --weights_dir=/data/ccy/fedproto/20260112_104826/weights --smooth_transition --output_dir=/data/ccy/fedproto/ --no_wandb"


    "python loop_df_fl.py --type=kd_train --iid=0 --epochs=80 --local_ep=1000 --lr=0.005 --batch_size 32 --synthesis_batch_size=8 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/siri-ours-accumuT-nosyn --model=res --dataset=siri-whu --adv=1 --beta=0.5 --seed=3407 --alg=ours --num_users=10 --checkpoint_steps 100 200 300 500 800 1300 2000 --weights_dir=/data/ccy/fedproto/20260112_104826/weights --output_dir=/data/ccy/fedproto/ --no_wandb"


    # Add more commands here...
)

# ================= Execution Loop =================
echo "=================================================================="
echo "Batch Runner Started"
echo "Target GPU: ${GPU_ID}"
echo "Total Commands: ${#CMDS[@]}"
echo "=================================================================="

count=1
total=${#CMDS[@]}

for cmd in "${CMDS[@]}"; do
    echo ""
    echo ">>> [${count}/${total}] Executing command:"
    echo "    ${cmd}"
    echo "------------------------------------------------------------------"
    
    # Run command with specified GPU
    CUDA_VISIBLE_DEVICES=${GPU_ID} eval "${cmd}"
    
    echo "------------------------------------------------------------------"
    echo ">>> Finished command [${count}/${total}]"
    count=$((count + 1))
done

echo ""
echo "=================================================================="
echo "All experiments completed successfully."
echo "=================================================================="
