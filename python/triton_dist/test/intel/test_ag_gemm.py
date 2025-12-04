import argparse
import os
import sys

import torch

from triton_dist.autotuner import contextual_autotune
from triton_dist.kernels.intel.allgather_gemm import ag_gemm, create_ag_gemm_context
from triton_dist.utils import (assert_allclose, group_profile)
from triton_dist.kernels.intel import initialize_distributed

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", default=False)
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="LLaMA-3.1-70B", choices=configs.keys())
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--local_world_size", default=8, type=int)

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


@register_test("correctness")
def test_ag_gemm(args, autotune=False):
    device = "xpu"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    M = 4091 * num_ranks
    N = 5120
    K = 1024

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    debug = args.debug
    ctx = create_ag_gemm_context(A, B, rank, num_ranks, num_local_ranks=args.local_world_size, max_M=M,
                                 for_correctness=debug)
    if rank == 0:
        print(f"all gather with: {ctx.all_gather_method}")

    def func():
        return ag_gemm(A, B, ctx=ctx, persistent=args.persistent, autotune=autotune)

    with group_profile("ag_gemm_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=args.default_group):
        for i in range(5):
            # every time, use a new input data to check correctness
            A.random_()
            B.random_()
            ctx.symm_workspace[:M].random_()
            C = func()

    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    for i in range(num_ranks):
        torch.distributed.barrier(args.default_group)
        if rank == i:
            print(f"Rank {rank}")
            assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)


register_test("correctness_autotune")(lambda args: test_ag_gemm(args, autotune=True))

configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}

if __name__ == "__main__":
    args = get_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.xpu.set_device(LOCAL_RANK)
    args.default_group = initialize_distributed()

    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    torch.distributed.destroy_process_group()
