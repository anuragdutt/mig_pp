import os
import gc
import json
import time
import queue
import logging
import traceback
import datetime
from typing import Dict, List, Optional, Tuple

from helpers import get_wiki_sample, load_specific_weights
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from transformers import DynamicCache, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import hub


import mig_transport_pipeline as mig_transport
import dcgm_mem_monitor as monitor
from token_exchange import TokenExchange

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

LOG_FILE = "benchmark.log"


def setup_logging(log_file: str = LOG_FILE) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="a")],
        force=True,
    )


log = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Vicuna-7B on a single 40GB A100 split into 3 MIG slices (20/10/10).
# fp16 weights ~13.5GB total, so even the 10GB slices have headroom for
# KV cache + activations at these sequence lengths.
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
TOTAL_LAYERS = 32
HIDDEN_SIZE = 4096
HEADS = 32

SEQ_LEN = 64
MAX_NEW_TOKENS = 512

MAX_RUNS = 10

# Restore the full sweep by uncommenting the other pairs and raising
# MAX_RUNS / MAX_NEW_TOKENS above.
BATCH_MB_PAIRS = [
    (24, 24), (24, 8), (24, 4)
]

# 3-slice: 3g.20gb + 2g.10gb + 2g.10gb
MIG_UUIDS = [
    "MIG-f06fb156-37b3-527c-8f93-23db507c9704",
    "MIG-814fba5b-2690-5fcf-be42-2f26f702dfc1",
    "MIG-7101d6b5-fe6e-594b-adab-2d6a259e8620",
    "MIG-e9631527-ed47-5561-8ba1-c6b599a5284c)2fc"
]

SLICE_GB = [20, 10, 5, 5]
LAYER_LIMITS = [22, 14, 7, 7]

WORLD_SIZE = len(MIG_UUIDS)

assert len(SLICE_GB) == WORLD_SIZE, "SLICE_GB must have one entry per rank"
assert len(LAYER_LIMITS) == WORLD_SIZE, "LAYER_LIMITS must have one entry per rank"
assert sum(LAYER_LIMITS) >= TOTAL_LAYERS, (
    f"LAYER_LIMITS sums to {sum(LAYER_LIMITS)} but the model has "
    f"{TOTAL_LAYERS} layers — no split can fit"
)

# Ordering constraint on splits. The two 10GB slices are identical, so
# permuting layers between them produces duplicate configurations with
# identical performance. Requiring l1 >= l2 breaks that symmetry and halves
# the search space. l0 > l1 keeps the biggest slice doing the most work,
# which is where the useful configurations live.
#
# Set to False to search the full space including "inverted" splits (small
# slice doing more work than a big one) — slower, and mostly OOMs.
ENFORCE_SLICE_ORDERING = True

# Dist message tag bases (avoid collisions)
PREFILL_TAG_BASE = 1000
DECODE_TAG_BASE = 2000
TOKENS_TAG_BASE = 9000


# This function was manually implemented to bypass a tensor shape mismatch
# caused by the default Hugging Face implementation:
# RuntimeError: The size of tensor a (40) must match the size of tensor b (128) at non-singleton dimension 3
def forward_through_layers(
    layers: nn.ModuleList,
    current_hidden: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    mask: Optional[torch.Tensor],
    cache: DynamicCache,
) -> torch.Tensor:
    """
    Passes the input tensor sequentially through a specific chunk of Transformer layers.

    Args:
        layers: A list of LLaMA Decoder layers assigned to this specific GPU.
        current_hidden: The input tensor containing the hidden states (activations) from the previous stage.
        position_embeddings: A tuple of (cos, sin) tensors for Rotary Position Embeddings (RoPE) so the model knows word order.
        mask: The attention mask (e.g., causal mask for prefill) to prevent words from looking into the future.
        cache: The Key-Value cache storing previous context to speed up text generation.

    Returns:
        torch.Tensor: The upgraded hidden states after passing through all assigned layers.
    """

    # Loop through every single layer assigned to this specific GPU
    for layer in layers:
        # 1. SAVE THE RESIDUAL (Skip Connection)
        # We keep an untouched copy of the data. If the complex math in this layer
        # degrades the signal, the network can fall back on this original copy.
        residual: torch.Tensor = current_hidden

        # 2. PRE-ATTENTION NORMALIZATION (RMSNorm)
        # Standardize the numbers to prevent them from growing too large and crashing the math.
        hidden_states: torch.Tensor = layer.input_layernorm(current_hidden)

        # 3. SELF-ATTENTION (The "Brain")
        # Words look at other words in the sequence to gather context and meaning.
        # It uses the cache to remember past words, and the mask to ignore future words.
        attn_outputs: Tuple[torch.Tensor, ...] = layer.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )

        # The self_attn function returns a tuple; the actual modified tensor is the first item [0].
        hidden_states = attn_outputs[0]

        # 4. FIRST MERGE
        # Add the new contextual insights (hidden_states) back into our untouched original copy (residual).
        hidden_states = residual + hidden_states

        # 5. SAVE NEW RESIDUAL
        # Update our "untouched copy" for the second half of the layer.
        residual = hidden_states

        # 6. PRE-MLP NORMALIZATION
        # Standardize the numbers again before the feed-forward network.
        hidden_states = layer.post_attention_layernorm(hidden_states)

        # 7. MULTI-LAYER PERCEPTRON / MLP (The "Muscle")
        # The AI processes the new context it just learned against its internal memorized weights.
        hidden_states = layer.mlp(hidden_states)

        # 8. FINAL MERGE
        # Add the MLP's output back into the residual to finalize this layer's upgrades.
        current_hidden = residual + hidden_states

    # Hand the fully processed box of data back to the pipeline so it can be shipped to the next GPU
    return current_hidden


# ---------------------------------------------------------------------------
# PIPELINE WORKER
# ---------------------------------------------------------------------------
def run_pipeline(
    rank: int,
    world_size: int,
    split_config: List[int],
    result_queue: mp.Queue,
    device_uuid: str,
    input_ids_seed: torch.Tensor,
    mb_size: int,
) -> None:
    setup_logging()

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=8),
        )

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # Transport logging -> logs/transport_<stamp>.log (shared by all ranks;
        # every line carries its rank, so grep 'rank2' to separate them).
        _tlog_path = mig_transport.setup_transport_logging()
        log.info(f"[Rank {rank}] transport log -> {_tlog_path}")

        # Patch dist send/recv to go through SHM+ACK transport.
        # Size the SHM/pinned slots to THIS run's mb_size rather than the
        # global worst case — every slot costs mb_size*SEQ_LEN*hidden*2 bytes
        # in both SHM and pinned RAM, on every rank. Slot count only needs to
        # cover the microbatches in flight, plus headroom for the batched
        # ACK drain.
        _num_mb = input_ids_seed.shape[0] // mb_size
        mig_transport.register_hooks(
            mb_size=mb_size,
            seq_len=SEQ_LEN,
            hidden_size=HIDDEN_SIZE,
            num_slots=_num_mb + 8,
        )

        # Getting model dimensions
        config = LlamaConfig.from_pretrained(MODEL_NAME)
        # Using Scaled Dot-Product Attention (Flash attention)
        config._attn_implementation = "sdpa"

        # Determing the start and end layer for each MIG instance
        start_layer = sum(split_config[:rank])
        end_layer = start_layer + split_config[rank]
        my_layer_indices = list(range(start_layer, end_layer))

        model_components: Dict[str, nn.Module] = {}

        # In a pipeline setup, Rank 0 is the very first GPU in the assembly line. It is the only worker that actually receives the raw input text from the user.
        # Because Ranks 1 and 2 only receive pre-processed mathematical data from the previous lockers,
        # they don't need to know how to translate raw text. By putting this inside an if rank == 0 block,
        # you prevent Ranks 1 and 2 from loading this translation dictionary into their memory, saving precious VRAM.
        if rank == 0:
            model_components["embed"] = (
                nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
            )

        # Registering layers with Torch
        layers = nn.ModuleList()
        for idx in my_layer_indices:
            layers.append(
                LlamaDecoderLayer(config, layer_idx=idx).half().to(device)
            )  # Empty physical layer

            # This is critical to do for 5gb instance
            torch.cuda.empty_cache()

        # Job of the last rank is the to translate raw data
        # Into english sentences
        if rank == world_size - 1:
            model_components["norm"] = (
                LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                .to(device)
                .half()
            )
            model_components["lm_head"] = (
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                .to(device)
                .half()
            )

        # Setting in inference mode
        for m in model_components.values():
            m.eval()
        layers.eval()

        # RoPE (Rotary Position Embeddings)
        rotary_embedding = LlamaRotaryEmbedding(config=config, device=device)

        # Weight loading reads every shard from disk and copies tensor by
        # tensor; on a cold page cache it dominates process startup and is the
        # reason a naive nsys capture is mostly this phase. Timed so the log
        # says so outright instead of leaving it to be inferred.
        _wl_t0 = time.perf_counter()
        load_specific_weights(
            rank, world_size, MODEL_NAME, layers, my_layer_indices, model_components
        )
        _wl_s = time.perf_counter() - _wl_t0
        log.info(
            f"[B01][Rank {rank}] weight load: {_wl_s:.1f}s "
            f"({len(my_layer_indices)} layers)"
        )

        # Deep clean after weight loading
        gc.collect()
        torch.cuda.empty_cache()

        input_ids = input_ids_seed.to(device)
        batch_size, seq_length = input_ids.shape

        if batch_size % mb_size != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by microbatch size {mb_size}"
            )
        num_microbatches = batch_size // mb_size

        # Creating distinct KV cache for each microbatch
        past_key_values_list = [DynamicCache() for _ in range(num_microbatches)]
        # waiting to hold the final predicted words for all batch_size amt of sentences.
        next_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # next_tokens is on the GPU but the process group is gloo, which is
        # CPU-only: it passes tensor.data_ptr() to writev(2), and a device
        # pointer there is EFAULT ("writev ...: Bad address"), killing the
        # rank and taking the others with it. Activations avoid this because
        # dist.isend is intercepted by the SHM engine and staged D2H;
        # patched_send/patched_recv deliberately bypass that for this small
        # control tensor, so the staging belongs here. TokenExchange owns one
        # pinned host buffer for the whole run and times every copy.
        # Verified by probe_gloo_send.py (T1-T7, 512-exchange decode loop).
        tok_exchange = TokenExchange(next_tokens)

        # Standard prefill mask code
        prefill_mask = torch.full(
            (1, 1, seq_length, seq_length),
            torch.finfo(torch.float16).min,
            device=device,
        )
        prefill_mask = torch.triu(prefill_mask, diagonal=1).half()

        # Pre-allocate recv buffers (reuse; don’t allocate per step)
        prefill_recv_bufs = None
        decode_recv_bufs = None

        # (the "Catching Mitts").
        # Pre-allocated so receiving actual data needs no fresh allocation.
        #
        # torch.empty, not torch.zeros: every receive overwrites the buffer in
        # full (mig_transport_pipeline.py:575 copies staging over the whole
        # tensor), so initial contents are never read. Zeroing here would also
        # be a default-stream write to a buffer that the transport later writes
        # from recv_stream with no cross-stream dependency — see the recv-buffer
        # note in CLAUDE.md. Allocating uninitialised removes that second writer
        # instead of trying to order it.
        if rank > 0:
            prefill_recv_bufs = [
                torch.empty(
                    (mb_size, seq_length, config.hidden_size),
                    dtype=torch.float16,
                    device=device,
                )
                for _ in range(num_microbatches)
            ]
            decode_recv_bufs = [
                torch.empty(
                    (mb_size, 1, config.hidden_size),
                    dtype=torch.float16,
                    device=device,
                )
                for _ in range(num_microbatches)
            ]

        dist.barrier()

        # Beginning of prefill loop
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Per-decode-step timing. The KV cache grows from seq_length to
        # seq_length + MAX_NEW_TOKENS over the run, so attention reads more
        # cache every step and per-step cost drifts upward — measured at
        # +10% on ACK wait and +43% on rank1 D2H block over 512 steps.
        # A single total_latency_ms averages that drift away, so record each
        # step separately and let the analysis see the curve.
        #
        # Events are pre-allocated: creating them inside the loop would add
        # allocation to the very thing being measured.
        step_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(MAX_NEW_TOKENS)
        ]
        step_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(MAX_NEW_TOKENS)
        ]

        # Per-microbatch COMPUTE timing (the layer stack only — no transport).
        # Nothing else measures this directly: the ACK wait [T11] reflects the
        # downstream rank's compute, but only indirectly and only as seen from
        # the sender. Measuring it here makes that inference checkable.
        #
        # One pair per (step, microbatch), flattened. Index: step_idx * n + mb.
        # Pre-allocated for the same reason as the per-step events, and read
        # back only after the final synchronize — calling elapsed_time() inside
        # the loop would force a sync and distort what it measures.
        _n_compute = (MAX_NEW_TOKENS + 1) * num_microbatches  # +1 for prefill
        compute_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(_n_compute)
        ]
        compute_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(_n_compute)
        ]

        # Prefill/decode phase split. total_latency_ms covers both, but they
        # are different regimes — prefill moves MB-scale activations and is
        # compute-bound in mb_size; decode moves KB and is bound by weight
        # streaming, which depends on microbatch COUNT and not size.
        prefill_end_event = torch.cuda.Event(enable_timing=True)

        # Barrier idle. Each barrier is a rendezvous, so a rank that arrives
        # early sits there. That idle time is pipeline imbalance and is
        # otherwise invisible — it is where a fast rank's time actually goes.
        barrier_wait_s = {"post_prefill": 0.0, "final": 0.0}

        start_event.record()

        with torch.no_grad():
            # Rank > 0: post *all* irecvs up front (max overlap)
            if rank > 0:
                prefill_recv_handles = [
                    dist.irecv(
                        prefill_recv_bufs[i],
                        src=rank - 1,
                        tag=PREFILL_TAG_BASE + i,
                    )
                    for i in range(num_microbatches)
                ]

            send_handles = []

            for mb_idx in range(num_microbatches):
                start_idx = mb_idx * mb_size
                end_idx = start_idx + mb_size
                tag = PREFILL_TAG_BASE + mb_idx

                if rank == 0:
                    current_hidden = model_components["embed"](
                        input_ids[start_idx:end_idx]
                    ).half()
                else:
                    # Just wait when you actually need MB(k)
                    prefill_recv_handles[mb_idx].wait()
                    current_hidden = prefill_recv_bufs[mb_idx]

                position_ids = torch.arange(
                    0, seq_length, dtype=torch.long, device=device
                ).unsqueeze(0)
                position_embeddings = rotary_embedding(current_hidden, position_ids)

                _ci = mb_idx  # prefill occupies slots [0, num_microbatches)
                compute_start_events[_ci].record()
                current_hidden = forward_through_layers(
                    layers,
                    current_hidden,
                    position_embeddings,
                    prefill_mask,
                    past_key_values_list[mb_idx],
                )
                compute_end_events[_ci].record()

                if rank == world_size - 1:
                    normed = model_components["norm"](current_hidden)
                    logits = model_components["lm_head"](normed[:, -1:, :])
                    next_tokens[start_idx:end_idx] = torch.argmax(logits, dim=-1)

                if rank < world_size - 1:
                    h = dist.isend(current_hidden.clone(), dst=rank + 1, tag=tag)
                    send_handles.append(h)

                    # Publish the PREVIOUS microbatch now: its D2H copy has
                    # had this microbatch's compute to run underneath, so
                    # flush() should find the event already complete and not
                    # block. Keeps the downstream rank fed one microbatch
                    # behind us instead of idling until our loop ends.
                    if len(send_handles) > 1:
                        send_handles[-2].flush()

            # Publish the last microbatch (nothing after it to hide behind).
            if send_handles:
                send_handles[-1].flush()

            # Drain ACKs and reclaim slots.
            for h in send_handles:
                h.wait()

            # Share next_tokens rank2 → rank0 (tagged)
            tok_tag = TOKENS_TAG_BASE + 0
            if rank == world_size - 1:
                tok_exchange.send(dst=0, tag=tok_tag)
            elif rank == 0:
                tok_exchange.recv(src=world_size - 1, tag=tok_tag)

            # Close prefill before the barrier so the phase split measures
            # prefill work, not prefill + waiting for the other ranks.
            prefill_end_event.record()

            _bt0 = time.perf_counter()
            dist.barrier()
            barrier_wait_s["post_prefill"] = time.perf_counter() - _bt0

            # =========================================================
            # DECODE — pipelined across microbatches
            # =========================================================
            for step in range(1, MAX_NEW_TOKENS + 1):
                step_start_events[step - 1].record()

                # Step boundary marker. The transport log interleaves all
                # ranks and encodes the step inside the message tag
                # (DECODE_TAG_BASE + step*10_000 + mb_idx), so segmenting it
                # by step otherwise means doing tag arithmetic. This makes
                # each step greppable directly.
                _tlog_step = logging.getLogger("mig_transport")
                _tlog_step.info(
                    "[T32][rank%d] ===== DECODE STEP %d/%d (n=%d) =====",
                    rank,
                    step,
                    MAX_NEW_TOKENS,
                    num_microbatches,
                )

                position_ids = (
                    torch.tensor(
                        [[seq_length + step - 1]], dtype=torch.long, device=device
                    )
                    .expand(mb_size, -1)
                    .contiguous()
                )

                # Rank > 0: reuse decode buffers, post *all* irecvs up front.
                #
                # No buf.zero_() here. It used to clear each buffer every step;
                # that was dead work (the H2D overwrites every element before
                # the sole reader below at `current_hidden = decode_recv_bufs`)
                # and it was a default-stream write racing the transport's
                # recv_stream write to the same memory. Removed rather than
                # synchronised — see CLAUDE.md, "recv buffers are not zeroed".
                if rank > 0:
                    decode_recv_handles = [
                        dist.irecv(
                            decode_recv_bufs[i],
                            src=rank - 1,
                            tag=DECODE_TAG_BASE + step * 10_000 + i,
                        )
                        for i in range(num_microbatches)
                    ]

                send_handles = []

                for mb_idx in range(num_microbatches):
                    start_idx = mb_idx * mb_size
                    end_idx = start_idx + mb_size
                    tag = DECODE_TAG_BASE + step * 10_000 + mb_idx

                    if rank == 0:
                        current_hidden = model_components["embed"](
                            next_tokens[start_idx:end_idx]
                        ).half()
                    else:
                        decode_recv_handles[mb_idx].wait()
                        current_hidden = decode_recv_bufs[mb_idx]

                    position_embeddings = rotary_embedding(current_hidden, position_ids)

                    _ci = step * num_microbatches + mb_idx
                    compute_start_events[_ci].record()
                    current_hidden = forward_through_layers(
                        layers,
                        current_hidden,
                        position_embeddings,
                        None,
                        past_key_values_list[mb_idx],
                    )
                    compute_end_events[_ci].record()

                    if rank == world_size - 1:
                        normed = model_components["norm"](current_hidden)
                        logits = model_components["lm_head"](normed)
                        next_tokens[start_idx:end_idx] = torch.argmax(logits, dim=-1)

                    if rank < world_size - 1:
                        h = dist.isend(current_hidden.clone(), dst=rank + 1, tag=tag)
                        send_handles.append(h)

                        # Publish previous microbatch (see prefill loop).
                        if len(send_handles) > 1:
                            send_handles[-2].flush()

                if send_handles:
                    send_handles[-1].flush()

                for h in send_handles:
                    h.wait()

                # next_tokens exchange (still a sync point; not a “bug”)
                if step < MAX_NEW_TOKENS:
                    tok_tag = TOKENS_TAG_BASE + step
                    if rank == world_size - 1:
                        tok_exchange.send(dst=0, tag=tok_tag)
                    elif rank == 0:
                        tok_exchange.recv(src=world_size - 1, tag=tok_tag)

                # Close the step AFTER the token exchange: that exchange is
                # part of the step's critical path (rank0 cannot start the
                # next step until it lands), so excluding it would understate
                # the step and hide the serialization.
                step_end_events[step - 1].record()

        _bt0 = time.perf_counter()
        dist.barrier()
        barrier_wait_s["final"] = time.perf_counter() - _bt0

        end_event.record()
        # REQUIRED, not removable, and deliberately full-device. Every
        # elapsed_time() call below reads CUDA events recorded across the
        # default stream AND the transport engine's copy_stream; querying an
        # event that has not completed raises or returns garbage. This is the
        # one place a whole-device drain is the correct tool.
        #
        # Costs nothing measurable: it runs once, after the final barrier,
        # outside the timed region. Not in the decode loop.
        torch.cuda.synchronize()

        total_latency_ms = start_event.elapsed_time(end_event)
        log.info(f"[Rank {rank}] Finished. Latency: {total_latency_ms:.0f} ms")

        # Read back per-step times. Safe here: the torch.cuda.synchronize()
        # above guarantees every recorded event has completed.
        step_latencies_ms = [
            step_start_events[i].elapsed_time(step_end_events[i])
            for i in range(MAX_NEW_TOKENS)
        ]
        if step_latencies_ms:
            _first = step_latencies_ms[0]
            _last = step_latencies_ms[-1]
            _drift = ((_last / _first) - 1.0) * 100.0 if _first > 0 else 0.0
            log.info(
                f"[B02][Rank {rank}] decode step latency: first={_first:.2f}ms "
                f"last={_last:.2f}ms drift={_drift:+.1f}% "
                f"mean={sum(step_latencies_ms) / len(step_latencies_ms):.2f}ms"
            )

        # Phase split: prefill vs decode.
        prefill_ms = start_event.elapsed_time(prefill_end_event)
        decode_ms = total_latency_ms - prefill_ms
        log.info(
            f"[B03][Rank {rank}] phase split: prefill={prefill_ms:.1f}ms "
            f"({prefill_ms / total_latency_ms * 100:.1f}%) "
            f"decode={decode_ms:.1f}ms "
            f"({decode_ms / total_latency_ms * 100:.1f}%)"
        )

        # Per-microbatch compute (layer stack only, no transport). Compare
        # against the downstream rank's [T11] ACK wait: if they match, the ACK
        # wait really is downstream compute rather than a transport cost.
        compute_ms = [
            compute_start_events[i].elapsed_time(compute_end_events[i])
            for i in range(_n_compute)
        ]
        _pre_compute = compute_ms[:num_microbatches]
        _dec_compute = compute_ms[num_microbatches:]
        if _pre_compute:
            log.info(
                f"[B04][Rank {rank}] prefill compute/mb: "
                f"mean={sum(_pre_compute) / len(_pre_compute):.2f}ms "
                f"total={sum(_pre_compute):.1f}ms over {len(_pre_compute)} mb"
            )
        if _dec_compute:
            _sorted = sorted(_dec_compute)
            log.info(
                f"[B05][Rank {rank}] decode compute/mb: "
                f"mean={sum(_dec_compute) / len(_dec_compute):.2f}ms "
                f"median={_sorted[len(_sorted) // 2]:.2f}ms "
                f"first={_dec_compute[0]:.2f}ms last={_dec_compute[-1]:.2f}ms "
                f"total={sum(_dec_compute) / 1000:.1f}s over {len(_dec_compute)} mb"
            )

        # Barrier idle — where a fast rank's time actually goes.
        log.info(
            f"[B06][Rank {rank}] barrier idle: "
            f"post_prefill={barrier_wait_s['post_prefill'] * 1000:.1f}ms "
            f"final={barrier_wait_s['final'] * 1000:.1f}ms"
        )

        # next_tokens staging cost — the price of routing the token exchange
        # around gloo's CPU-only transport. Reported on its own line rather
        # than folded into the step, so it stays checkable: measured at
        # ~0.04ms per exchange against ~745ms ACK waits, i.e. noise. If this
        # ever stops being noise, it will be visible here first.
        log.info(f"[B07][Rank {rank}] next_tokens staging: {tok_exchange.summary()}")

        # Per-rank transport verdict: did the async copies actually overlap?
        # Grep the log for T28 to get one VERDICT line per rank.
        mig_transport.log_summary(
            label=f"split={split_config} mb={mb_size} latency={total_latency_ms:.0f}ms"
        )

        if rank == 0:
            result_queue.put(("latency", total_latency_ms))

        # Every rank reports its own curve — the stages are asymmetric (the
        # bottleneck rank is the one that sets the step time), so rank0 alone
        # would not show where the drift comes from.
        result_queue.put((f"step_latencies_rank{rank}", step_latencies_ms))
        result_queue.put(
            (
                f"phase_rank{rank}",
                {
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "weight_load_s": _wl_s,
                    "barrier_post_prefill_ms": barrier_wait_s["post_prefill"] * 1000,
                    "barrier_final_ms": barrier_wait_s["final"] * 1000,
                    "decode_compute_mean_ms": (
                        sum(_dec_compute) / len(_dec_compute) if _dec_compute else 0.0
                    ),
                    "prefill_compute_mean_ms": (
                        sum(_pre_compute) / len(_pre_compute) if _pre_compute else 0.0
                    ),
                },
            )
        )

        dist.destroy_process_group()

        # Release SHM segments. Without this each run leaves NUM_SLOTS
        # segments per rank in /dev/shm; over a long sweep they accumulate
        # until allocation fails.
        mig_transport.cleanup()

    except torch.cuda.OutOfMemoryError:
        log.error(f"[Rank {rank}] OOM")
        result_queue.put(("oom", rank))
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        try:
            mig_transport.cleanup()
        except Exception:
            pass

    except Exception:
        log.error(f"[Rank {rank}] Unexpected exception:\n{traceback.format_exc()}")
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        try:
            mig_transport.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SPLIT GENERATION
# ---------------------------------------------------------------------------


def generate_layer_splits():
    """
    Enumerate every way to distribute TOTAL_LAYERS across the MIG slices,
    subject to per-slice capacity (LAYER_LIMITS) and the symmetry-breaking
    ordering constraint.

    Works for any WORLD_SIZE: recurses over ranks 0..n-2 and lets the last
    rank take the remainder. Ordering rules come from SLICE_GB — see
    ENFORCE_SLICE_ORDERING above.
    """
    valid_splits = []
    n = WORLD_SIZE

    def _ok(prev_idx, prev_layers, layers):
        """Ordering constraint between rank prev_idx and the next rank."""
        if not ENFORCE_SLICE_ORDERING:
            return True
        if SLICE_GB[prev_idx] == SLICE_GB[prev_idx + 1]:
            return prev_layers >= layers
        return prev_layers > layers

    def _recurse(rank, assigned, remaining):
        # Last rank takes whatever is left — no need to enumerate it.
        if rank == n - 1:
            if not (1 <= remaining <= LAYER_LIMITS[rank]):
                return
            if assigned and not _ok(rank - 1, assigned[-1], remaining):
                return
            valid_splits.append(assigned + [remaining])
            return

        # Leave at least one layer for each rank still to come.
        max_here = min(LAYER_LIMITS[rank], remaining - (n - rank - 1))
        for count in range(1, max_here + 1):
            if assigned and not _ok(rank - 1, assigned[-1], count):
                continue
            _recurse(rank + 1, assigned + [count], remaining - count)

    _recurse(0, [], TOTAL_LAYERS)
    return valid_splits


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    setup_logging()

    # One shared timestamp for this whole sweep, inherited by every spawned
    # rank via the environment, so all ranks write into the same
    # logs/transport_<stamp>.log instead of four separate files.
    os.environ.setdefault("MIG_LOG_STAMP", time.strftime("%Y%m%d_%H%M%S"))
    log.info(f"Transport log stamp: {os.environ['MIG_LOG_STAMP']}")

    log.info("Setting up DCGM monitor group...")
    # Pass the topology so the monitor can map DCGM entities to the right
    # ranks by MIG UUID, rather than a hardcoded entity->rank table.
    monitor.setup_dcgm_group(mig_uuids=MIG_UUIDS, slice_gb=SLICE_GB)

    selected_splits = generate_layer_splits()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Flatten the sweep into an explicit list so MAX_RUNS can cap it cleanly.
    run_plan = [
        (split, batch_size, mb_size)
        for split in selected_splits
        for batch_size, mb_size in BATCH_MB_PAIRS
    ]
    full_sweep_size = len(run_plan)

    if MAX_RUNS is not None and full_sweep_size > MAX_RUNS:
        run_plan = run_plan[:MAX_RUNS]
        log.warning(
            f"MAX_RUNS={MAX_RUNS} is capping this sweep: running {len(run_plan)} "
            f"of {full_sweep_size} configurations. Set MAX_RUNS=None for the "
            f"full sweep."
        )

    total_runs = len(run_plan)
    results = []
    # Per-decode-step latencies, one row per (config, rank, step). Kept out of
    # the main results CSV because it is MAX_NEW_TOKENS * WORLD_SIZE rows per
    # configuration — the summary table stays readable, the curve lives here.
    step_latency_rows = []
    current_run = 1

    for split, batch_size, mb_size in run_plan:
        log.info(
            f"[{current_run}/{total_runs}] Split {split} | "
            f"Batch: {batch_size} | Microbatch: {mb_size} | "
            f"Microbatches: {batch_size // mb_size}"
        )

        input_ids_seed = get_wiki_sample(batch_size, SEQ_LEN, MODEL_NAME)

        label = f"s{'_'.join(map(str, split))}_b{batch_size}_mb{mb_size}"
        monitor.set_label(label)
        monitor.clear()
        monitor.start()

        q = mp.Queue()
        procs: list[mp.Process] = []

        try:
            for rank in range(WORLD_SIZE):
                p = mp.Process(
                    target=run_pipeline,
                    args=(
                        rank,
                        WORLD_SIZE,
                        split,
                        q,
                        MIG_UUIDS[rank],
                        input_ids_seed,
                        mb_size,
                    ),
                )
                p.start()
                procs.append(p)

            JOIN_TIMEOUT_S = 1200
            for p in procs:
                p.join(timeout=JOIN_TIMEOUT_S)

            hung = [p for p in procs if p.is_alive()]
            if hung:
                log.error(
                    "Hang detected: ranks still alive after timeout: "
                    + ", ".join(str(procs.index(p)) for p in hung)
                )
                for p in hung:
                    p.terminate()
                for p in hung:
                    p.join(timeout=10)

            monitor.stop()

            # Sample row: (timestamp, label, gpu_mb, gi0_mb, gi1_mb, ...)
            # One gi<N>_mb column per MIG slice, in MIG_UUIDS order.
            run_samples = list(monitor._samples)
            # print("rum samples", run_samples)
            peak_per_rank = [
                max((row[3 + r] for row in run_samples), default=0)
                for r in range(WORLD_SIZE)
            ]
            avg_per_rank = [
                (
                    (sum(row[3 + r] for row in run_samples) / len(run_samples))
                    if run_samples
                    else 0
                )
                for r in range(WORLD_SIZE)
            ]

            exit_codes = [p.exitcode for p in procs]
            queue_items = {}
            try:
                while True:
                    key, val = q.get(timeout=2.0)
                    queue_items[key] = val
            except queue.Empty:
                pass

            # Harvest per-step curves before branching on run outcome, so a
            # partial run still yields whatever steps completed.
            for _r in range(WORLD_SIZE):
                for _step_idx, _ms in enumerate(
                    queue_items.get(f"step_latencies_rank{_r}", []), start=1
                ):
                    step_latency_rows.append(
                        {
                            "split": str(split),
                            "batch_size": batch_size,
                            "microbatch_size": mb_size,
                            "num_microbatches": batch_size // mb_size,
                            "rank": _r,
                            "step": _step_idx,
                            "step_latency_ms": _ms,
                        }
                    )

            base_row = {
                "split": str(split),
                "batch_size": batch_size,
                "microbatch_size": mb_size,
                "num_microbatches": batch_size // mb_size,
                "max_new_tokens": MAX_NEW_TOKENS,
                # Per-slice memory columns, generated from SLICE_GB so
                # they stay correct when the topology changes.
                **{
                    f"peak_rank{r}_{SLICE_GB[r]}gb_mb": peak_per_rank[r]
                    for r in range(WORLD_SIZE)
                },
                **{
                    f"avg_rank{r}_{SLICE_GB[r]}gb_mb": round(avg_per_rank[r])
                    for r in range(WORLD_SIZE)
                },
                "total_latency_ms": None,
                "status": None,
                # Phase/compute breakdown from the bottleneck-visible ranks.
                # Per-rank so an imbalanced split is visible in the summary
                # table rather than only in the transport log.
                **{
                    f"prefill_ms_rank{r}": queue_items.get(f"phase_rank{r}", {}).get(
                        "prefill_ms"
                    )
                    for r in range(WORLD_SIZE)
                },
                **{
                    f"decode_ms_rank{r}": queue_items.get(f"phase_rank{r}", {}).get(
                        "decode_ms"
                    )
                    for r in range(WORLD_SIZE)
                },
                **{
                    f"decode_compute_mean_ms_rank{r}": queue_items.get(
                        f"phase_rank{r}", {}
                    ).get("decode_compute_mean_ms")
                    for r in range(WORLD_SIZE)
                },
                **{
                    f"barrier_idle_ms_rank{r}": (
                        queue_items.get(f"phase_rank{r}", {}).get(
                            "barrier_post_prefill_ms", 0.0
                        )
                        or 0.0
                    )
                    + (
                        queue_items.get(f"phase_rank{r}", {}).get(
                            "barrier_final_ms", 0.0
                        )
                        or 0.0
                    )
                    for r in range(WORLD_SIZE)
                },
                **{
                    f"weight_load_s_rank{r}": queue_items.get(
                        f"phase_rank{r}", {}
                    ).get("weight_load_s")
                    for r in range(WORLD_SIZE)
                },
            }

            if hung:
                base_row["status"] = "hang"

            elif "oom" in queue_items:
                oom_rank = queue_items["oom"]
                log.warning(f"OOM on Rank {oom_rank} — skipping.")
                base_row["status"] = f"OOM_rank{oom_rank}"

            elif any((code is not None) and (code != 0) for code in exit_codes):
                log.error(f"Crashed. Exit codes: {exit_codes}")
                base_row["status"] = "crash"

            elif "latency" in queue_items:
                latency = queue_items["latency"]
                log.info(f"Total latency:     {latency:.0f} ms")
                # Report the tightest slice — that's the one at risk of OOM.
                # (Was hardcoded to index 3, which IndexErrors on any
                # topology with fewer than 4 slices.)
                _tight = min(range(WORLD_SIZE), key=lambda r: SLICE_GB[r])
                log.info(
                    f"Peak {SLICE_GB[_tight]}GB memory (rank {_tight}): "
                    f"{peak_per_rank[_tight]} MB"
                )
                log.info(
                    f"Avg  {SLICE_GB[_tight]}GB memory (rank {_tight}): "
                    f"{avg_per_rank[_tight]:.0f} MB"
                )
                base_row["total_latency_ms"] = latency
                base_row["status"] = "ok"

            else:
                log.warning("Timeout — no results received.")
                base_row["status"] = "timeout"

            results.append(base_row)

        finally:
            try:
                monitor.stop()
            except Exception:
                pass
            q.close()
            q.join_thread()

        for p in procs:
            if p.is_alive():
                p.terminate()

        current_run += 1

    df = pd.DataFrame(results)
    df.to_csv("mig_benchmark_results.csv", index=False)

    # Per-step decode curves. One row per (config, rank, step) — the KV cache
    # grows over a run, so step cost drifts and a single mean hides it.
    if step_latency_rows:
        step_df = pd.DataFrame(step_latency_rows)
        step_df.to_csv("mig_step_latencies.csv", index=False)
        log.info(
            f"Wrote {len(step_latency_rows)} per-step rows -> mig_step_latencies.csv"
        )

    monitor.save_csv("mig_memory_trace.csv")

    successful = df[df["status"] == "ok"]
    if not successful.empty:
        # Column names follow the SLICE_GB topology, so build them here
        # rather than hardcoding rank2/rank3.
        peak_cols = [f"peak_rank{r}_{SLICE_GB[r]}gb_mb" for r in range(WORLD_SIZE)]

        def _peaks(row):
            return " | ".join(
                f"R{r}({SLICE_GB[r]}GB): {row[peak_cols[r]]} MB"
                for r in range(WORLD_SIZE)
            )

        log.info("--- Best by Latency ---")
        best_lat = successful.loc[successful["total_latency_ms"].idxmin()]
        log.info(
            f"Split: {best_lat['split']} | Batch: {best_lat['batch_size']} "
            f"| MB: {best_lat['microbatch_size']} "
            f"| Latency: {best_lat['total_latency_ms']:.0f} ms "
            f"| {_peaks(best_lat)}"
        )

        # Most memory efficient on the tightest slice, at the largest batch
        # that actually produced results.
        tightest = min(range(WORLD_SIZE), key=lambda r: SLICE_GB[r])
        tightest_col = peak_cols[tightest]

        largest_batch = successful["batch_size"].max()
        biggest = successful[successful["batch_size"] == largest_batch]

        if not biggest.empty:
            log.info(
                f"--- Most Memory Efficient (lowest peak on the "
                f"{SLICE_GB[tightest]}GB slice at batch={largest_batch}) ---"
            )
            best_mem = biggest.loc[biggest[tightest_col].idxmin()]
            log.info(
                f"Split: {best_mem['split']} | MB: {best_mem['microbatch_size']} "
                f"| Latency: {best_mem['total_latency_ms']:.0f} ms "
                f"| {_peaks(best_mem)}"
            )

        oom_count = len(df[df["status"].str.startswith("OOM", na=False)])
        log.info(f"{oom_count} configurations skipped due to OOM")

    log.info("Done.")
    log.info("Benchmark results → mig_benchmark_results.csv")
    log.info("Per-step decode   → mig_step_latencies.csv")
    log.info("Memory trace      → mig_memory_trace.csv")


if __name__ == "__main__":
    main()
