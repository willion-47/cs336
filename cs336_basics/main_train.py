import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW, clip_gradient_norm
from cs336_basics.scheduler import get_lr_cosine_schedule


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize_device(device: str) -> None:
    """Wait for asynchronous device work before reading a wall-clock timer."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def load_token_data(path: str, dtype: str, context_length: int) -> np.memmap:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Token data not found: {data_path}")

    np_dtype = np.dtype(dtype)
    if data_path.stat().st_size % np_dtype.itemsize != 0:
        raise ValueError(
            f"{data_path} is not aligned to dtype={dtype} "
            f"({np_dtype.itemsize} bytes per token)"
        )

    data = np.memmap(data_path, dtype=np_dtype, mode="r")
    if len(data) <= context_length:
        raise ValueError(
            f"{data_path} only contains {len(data)} tokens, but context_length="
            f"{context_length} requires at least {context_length + 1}"
        )
    return data


@torch.no_grad()
def estimate_validation_loss(
    model: TransformerLM,
    data: np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    return float(np.mean(losses))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CS336 Assignment 1 Transformer LM")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--no_rms_norm", action="store_true")
    parser.add_argument("--norm_mode", choices=["pre", "post"], default="pre")
    parser.add_argument("--no_rope", action="store_true")
    parser.add_argument("--ffn_type", choices=["swiglu", "silu"], default="swiglu")

    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--valid_data_path", required=True)
    parser.add_argument(
        "--data_dtype",
        choices=["uint16", "uint32", "int32", "int64"],
        default="uint16",
        help="Must match the dtype used by preprocess.py",
    )
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from OUT_DIR/ckpt.pt when it exists (use --no-resume for a fresh run)",
    )
    parser.add_argument("--resume_from", default=None, help="Explicit checkpoint path to resume from")

    parser.add_argument("--wandb_project", default="cs336-pretraining")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
        help="Logging is disabled by default so local training never requires a WandB login",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if not args.no_rope and (args.d_model // args.num_heads) % 2 != 0:
        raise ValueError("RoPE requires an even head dimension (d_model / num_heads)")
    if args.max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    if not 0 <= args.warmup_iters < args.max_iters:
        raise ValueError("warmup_iters must satisfy 0 <= warmup_iters < max_iters")
    if args.eval_iters < 1 or args.log_interval < 1 or args.checkpoint_interval < 1:
        raise ValueError("eval_iters, log_interval, and checkpoint_interval must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    args.device = device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_token_data(args.train_data_path, args.data_dtype, args.context_length)
    val_data = load_token_data(args.valid_data_path, args.data_dtype, args.context_length)
    print(f"device={device} | train={len(train_data):,} tokens | valid={len(val_data):,} tokens")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=None if args.no_rope else args.rope_theta,
        device=device,
        use_rms_norm=not args.no_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = Path(args.resume_from) if args.resume_from else out_dir / "ckpt.pt"
    start_iter = 0
    if args.resume_from and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if (args.resume_from or args.resume) and checkpoint_path.is_file():
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"resumed from {checkpoint_path} at completed iteration {start_iter}")

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            mode=args.wandb_mode,
        )

    completed_iters = start_iter
    synchronize_device(device)
    throughput_window_start = time.perf_counter()
    throughput_window_tokens = 0
    throughput_window_steps = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for it in range(start_iter, args.max_iters):
            lr = get_lr_cosine_schedule(
                it,
                args.lr,
                args.min_lr,
                args.warmup_iters,
                args.max_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.train()
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            batch_max_token_id = max(x.max().item(), y.max().item())
            if batch_max_token_id >= args.vocab_size:
                raise ValueError(
                    f"Batch contains token ID {batch_max_token_id}, but vocab_size="
                    f"{args.vocab_size}. Use the same tokenizer vocabulary for "
                    "preprocessing and training."
                )
            loss = cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {it}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradient_norm(model.parameters(), args.max_norm)
            optimizer.step()
            completed_iters = it + 1
            throughput_window_tokens += x.numel()
            throughput_window_steps += 1

            should_log = it == start_iter or completed_iters % args.log_interval == 0 or completed_iters == args.max_iters
            if should_log:
                # CUDA/MPS operations are asynchronous. Synchronize only at log
                # boundaries so timing stays accurate without slowing every step.
                synchronize_device(device)
                training_seconds = time.perf_counter() - throughput_window_start
                tokens_per_second = throughput_window_tokens / max(training_seconds, 1e-12)
                step_time_ms = 1000.0 * training_seconds / throughput_window_steps
                tokens_seen = completed_iters * args.batch_size * args.context_length

                val_loss = estimate_validation_loss(
                    model,
                    val_data,
                    args.batch_size,
                    args.context_length,
                    device,
                    args.eval_iters,
                )
                metrics = {
                    "train/loss": loss.item(),
                    "val/loss": val_loss,
                    "lr": lr,
                    "iter": completed_iters,
                    "train/tokens_per_second": tokens_per_second,
                    "train/step_time_ms": step_time_ms,
                    "train/tokens_seen": tokens_seen,
                }
                if device.startswith("cuda"):
                    metrics["train/peak_gpu_memory_gb"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                print(
                    f"iter {completed_iters:>6}/{args.max_iters} | "
                    f"train {loss.item():.4f} | valid {val_loss:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_second:,.0f} tok/s"
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed_iters)

            if completed_iters % args.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, completed_iters, out_dir / "ckpt.pt")

            if should_log:
                # Start the next timing window after validation/logging/checkpoint
                # so tokens/s measures training rather than bookkeeping work.
                synchronize_device(device)
                throughput_window_start = time.perf_counter()
                throughput_window_tokens = 0
                throughput_window_steps = 0
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, completed_iters, out_dir / "ckpt.pt")
        print(f"interrupted; saved checkpoint at iteration {completed_iters}")
        return
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    save_checkpoint(model, optimizer, completed_iters, out_dir / "ckpt.pt")
    save_checkpoint(model, optimizer, completed_iters, out_dir / "ckpt_final.pt")
    print(f"training complete; checkpoint saved to {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
