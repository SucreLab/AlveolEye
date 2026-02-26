export NCCL_CUMEM_ENABLE=0
export OMP_NUM_THREADS=6
torchrun --nproc-per-node=3 ./src/alveoleye/paper_scripts/optimal_training_size.py