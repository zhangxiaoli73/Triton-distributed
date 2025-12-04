
export PYTHONPATH=/home/gta/cherry/Triton-distributed/python:$PYTHONPATH

TRITON_DIST_LOG_LEVEL=info torchrun  --nproc_per_node=2 --nnodes=1 --rdzv_endpoint=127.0.0.1:23456 --local_ranks_filter=0,1 test_ag_gemm.py --case correctness --local_world_size=2


