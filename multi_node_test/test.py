import torch
import os
import torch.distributed as dist

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.environ.get("RANK", 0))


if __name__ == "__main__":
    print("Initializing process group...")
    dist.init_process_group("nccl")
    print(f"WORLD_SIZE: {WORLD_SIZE}, LOCAL_RANK: {LOCAL_RANK}, GLOBAL_RANK: {GLOBAL_RANK}")

    if GLOBAL_RANK == 0:
        A = torch.ones(10, 10).to("cuda") * 5
        B = torch.ones(10, 10).to("cuda") * 6
        scalar = torch.tensor([12.0]).to("cuda")
        A_chunks = [A[:5], A[5:]]
        B_chunks = [B[:5], B[5:]]
        scalar_chunks = [scalar, scalar]
        A_local = torch.zeros(5, 10, device=f"cuda:{LOCAL_RANK}")
        B_local = torch.zeros(5, 10, device=f"cuda:{LOCAL_RANK}")
        scalar_local = torch.zeros(1, device=f"cuda:{LOCAL_RANK}")
        
        torch.distributed.scatter(A_local, A_chunks, src=0)
        torch.distributed.scatter(B_local, B_chunks, src=0)
        torch.distributed.scatter(scalar_local, scalar_chunks, src=0)
    else:
        A_local = torch.zeros(5, 10, device=f"cuda:{LOCAL_RANK}")
        B_local = torch.zeros(5, 10, device=f"cuda:{LOCAL_RANK}")
        scalar_local = torch.zeros(1, device=f"cuda:{LOCAL_RANK}")
        
        torch.distributed.scatter(A_local, None, src=0)
        torch.distributed.scatter(B_local, None, src=0)
        torch.distributed.scatter(scalar_local, None, src=0)
    
    local_result = torch.addcmul(A_local, B_local, scalar_local)
    
    if GLOBAL_RANK == 0:
        result = torch.zeros(10, 10, device=f"cuda:{LOCAL_RANK}")
        result_chunks = [torch.zeros(5, 10, device=f"cuda:{LOCAL_RANK}") for _ in range(WORLD_SIZE)]
        torch.distributed.gather(local_result, result_chunks, dst=0)
        result[:5] = result_chunks[0]
        result[5:] = result_chunks[1]
        print(f"Result: {result}")
    else:
        torch.distributed.gather(local_result, None, dst=0)
    dist.destroy_process_group()
