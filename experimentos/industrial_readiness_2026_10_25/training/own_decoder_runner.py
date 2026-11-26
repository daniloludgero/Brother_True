"""Colab-only pilot for the own decoder, initialized from scratch.

This is an engineering SFT pilot, not language pretraining or commercial
qualification. Sixty instruction pairs cannot establish language competence of
a random decoder. No teacher weights are loaded. No resume is offered yet.

Each reviewed JSONL row contains id, split, prompt or messages, response,
bio_vector20, bio_provenance and evidence. teacher_review.payload_sha256 covers
all fields except teacher_review/review_status (canonical JSON, sorted keys).
Evidence of teacher execution must contain the same digest and approved=true.
Core traces contain vector20, energy_crit, collapse_checked=true and
prompt_sha256=canonical_hash(messages_for(row)), and match the declared SHA-256.
Hashes bind data identity; they do not prove that a core or teacher was executed.
Evaluation rows omit the supervised response; any reference fields are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
EXPERIMENT = "industrial_readiness_2026_10_25"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def reviewed_payload(row):
    return {key: value for key, value in row.items()
            if key not in ("teacher_review", "review_status")}


def atomic_json(path, payload):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cloud_guard(run_dir):
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("COLAB ONLY: no training, inference, or downloads on host")
    import google.colab  # noqa: F401
    if not os.path.ismount("/content/drive"):
        raise RuntimeError("Mount Google Drive before invoking the runner")
    base = Path("/content/drive/MyDrive/brothertrue/experiments") / EXPERIMENT
    run = Path(run_dir).resolve()
    if base.resolve() not in run.parents:
        raise ValueError("run-dir must be a new child of the isolated experiment on Drive")
    if run.exists():
        raise FileExistsError("Use a new run-dir; prior results are immutable")
    return run


def cloud_file(path):
    path = Path(path).resolve()
    if Path("/content") not in path.parents or not path.is_file():
        raise ValueError(f"Existing cloud file required: {path}")
    return path


def read_rows(path, approved=False):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if len(line) > 131072:
                raise ValueError("Pilot JSONL line exceeds 128 KiB")
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError("Each row requires a string id")
            if approved:
                review = row.get("teacher_review", {})
                digest = canonical_hash(reviewed_payload(row))
                proof = cloud_file(review.get("evidence", ""))
                if (review.get("status") != "approved" or not review.get("reviewer")
                    or review.get("payload_sha256") != digest
                    or sha256(proof) != review.get("evidence_sha256")):
                    raise ValueError("Missing or altered review of the exact complete row")
                evidence = json.loads(proof.read_text(encoding="utf-8"))
                if evidence.get("payload_sha256") != digest or evidence.get("approved") is not True:
                    raise ValueError("Teacher execution evidence does not approve this row")
                if row.get("split") not in ("train", "validation"):
                    raise ValueError("SFT accepts only assigned train/validation rows")
                if not isinstance(row.get("response"), str) or not row["response"].strip():
                    raise ValueError("A nonempty supervised response is required")
            rows.append(row)
            if len(rows) > 1024:
                raise ValueError("Pilot bounded to 1024 rows; use separate streaming shards")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty data with unique ids required")
    return rows


def messages_for(row):
    messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Messages must end with user, excluding target response")
    if any(m.get("role") not in ("system", "user", "assistant")
           or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Invalid messages")
    return messages


def next_token_targets(prompt, target, max_length):
    """Mask prompt-only labels while retaining the first response token and EOS."""
    if not prompt or not target or len(prompt) + len(target) - 1 > max_length:
        raise ValueError("Empty or oversized supervised sequence; no silent truncation")
    return (prompt + target)[:-1], [-100] * (len(prompt) - 1) + target


def validate_core_trace(row):
    provenance = row.get("bio_provenance", {})
    trace_path = cloud_file(provenance.get("trace_path", ""))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if (provenance.get("producer") != "brothertrue_core"
        or sha256(trace_path) != provenance.get("trace_sha256")
        or trace.get("vector20") != row.get("bio_vector20")
        or trace.get("energy_crit") != provenance.get("energy_crit")
        or trace.get("collapse_checked") is not True
        or provenance.get("collapse_checked") is not True
        or trace.get("prompt_sha256") != canonical_hash(messages_for(row))):
        raise ValueError("Exact prompt-bound isolated core trace required")
    return trace_path, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("sft", "evaluate"))
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-license", default="unverified")
    parser.add_argument("--data", required=True)
    parser.add_argument("--heldout")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--condition", choices=("full", "without_bio", "without_evidence"), default="full")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    if min(args.max_steps, args.accumulation, args.save_every, args.max_length,
           args.max_new_tokens, args.d_model, args.layers) <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Positive finite limits required")
    run = cloud_guard(args.run_dir)
    root = Path(args.project_root).resolve()
    if Path("/content") not in root.parents:
        raise ValueError("project-root must be the isolated cloud source bundle")
    exp = root / "experimentos" / EXPERIMENT
    sys.path[:0] = [str(exp), str(root)]
    from contracts import (BiocognitiveState, Citation, EvidenceClaim, EvidenceDecision,
                           EvidenceSource, ThoughtRenderer, build_response)
    import numpy as np
    import psutil
    import torch
    from tokenizers import Tokenizer
    from modeling.own_voice import EVIDENCE_ORDER, OwnVoice
    from resource_governor import ResourceGovernor, ResourcePolicy
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU training fallback")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75, ram_limit_percent=60))

    def throttle():
        governor.throttle()
        if psutil.cpu_percent(interval=0.1) > 75 or psutil.virtual_memory().percent > 60:
            raise RuntimeError("Cloud CPU/RAM policy exceeded; stop cooperatively")
        for location in (Path("/content"), run.parent):
            existing = location
            while not existing.exists():
                existing = existing.parent
            if shutil.disk_usage(existing).free < 4.5 * 1024**3:
                raise RuntimeError("Cloud disk below 4.5 GiB; no automatic deletion")

    throttle()
    tok_path, data_path = cloud_file(args.tokenizer), cloud_file(args.data)
    rows = read_rows(data_path, approved=args.mode == "sft")
    tokenizer = Tokenizer.from_file(str(tok_path))
    special = {name: tokenizer.token_to_id(f"<{name}>") for name in ("pad", "bos", "eos")}
    if any(v is None for v in special.values()) or len(set(special.values())) != 3:
        raise ValueError("Distinct pad/bos/eos tokens required in the pinned tokenizer")

    def prepare(row):
        trace_path, provenance = validate_core_trace(row)
        state = BiocognitiveState.from_core(row["bio_vector20"], source=str(trace_path),
            energy_crit=provenance["energy_crit"], collapse_checked=provenance["collapse_checked"])
        evidence = row.get("evidence", {})
        sources = tuple(EvidenceSource(**s) for s in evidence.get("sources", []))
        claims = tuple(EvidenceClaim(**{**c, "citations": tuple(Citation(**q)
            for q in c.get("citations", []))}) for c in evidence.get("claims", []))
        decision = EvidenceDecision(evidence.get("mode", "grounded"), claims, sources)
        # Raw retrieved prose is a delimited JSON data field, never the control channel.
        context = [{"id": s.id, "text": s.text} for s in sources]
        prompt = ("[dialogue]\n" + json.dumps(messages_for(row), ensure_ascii=False)
            + "\n[retrieved_data]\n" + json.dumps(context, ensure_ascii=False)
            + "\n" + ThoughtRenderer.render(state, decision) + "\n[response]\n")
        ids = [special["bos"]] + tokenizer.encode(prompt, add_special_tokens=False).ids
        if any(token in set(special.values()) for token in ids[1:]):
            raise ValueError("Prompt data contains reserved BOS/EOS/padding tokens")
        bio = torch.tensor([state.vector20], device="cuda", dtype=torch.float32)
        index = {"supported": 0, "partial": 1, "insufficient": 2, "conflicting": 3}[decision.verdict]
        if not claims:
            index = EVIDENCE_ORDER.index("unassessed")
        features = torch.zeros((1, 5), device="cuda")
        features[0, index] = 1
        return state, decision, ids, bio, features

    prepared = {row["id"]: prepare(row) for row in rows}
    if args.mode == "sft":
        if args.checkpoint or args.condition != "full" or not args.heldout:
            raise ValueError("SFT is fresh from-scratch/full; heldout separation required; no resume")
        heldout_path = cloud_file(args.heldout)
        heldout = read_rows(heldout_path)
        def key(r):
            return canonical_hash([{"role": m["role"], "content": " ".join(m["content"].split()).casefold()}
                                   for m in messages_for(r)])
        train, validation = ([r for r in rows if r["split"] == split] for split in ("train", "validation"))
        if not train or not validation or {key(r) for r in train} & {key(r) for r in validation}:
            raise ValueError("Disjoint nonempty training/validation sets required")
        if {key(r) for r in rows} & {key(r) for r in heldout}:
            raise ValueError("Training/heldout prompt overlap")
    elif not args.checkpoint:
        raise ValueError("Evaluation requires a completed own-decoder checkpoint")
    config = dict(vocab_size=tokenizer.get_vocab_size(), d_model=args.d_model, nhead=4,
        num_layers=args.layers, dim_feedforward=args.d_model * 3, max_len=args.max_length,
        pad_token_id=special["pad"], bos_token_id=special["bos"], eos_token_id=special["eos"])
    checkpoint_identity = None
    if args.checkpoint:
        checkpoint = cloud_file(args.checkpoint)
        manifest = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256(checkpoint) != manifest["checkpoint_sha256"] or sha256(tok_path) != manifest["tokenizer_sha256"]:
            raise ValueError("Checkpoint/tokenizer identity mismatch")
        config = manifest["model_config"]
        checkpoint_identity = manifest["checkpoint_sha256"]
    model = OwnVoice(**config).to("cuda")
    if args.checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["architecture_version"] != model.architecture_version:
            raise ValueError("Architecture mismatch")
        model.load_state_dict(payload["model"], strict=True)
    run.mkdir(parents=True)
    identity = {"architecture_version": model.architecture_version,
        "initialization": "from_scratch" if args.mode == "sft" else "own_checkpoint",
        "teacher_weights_loaded": False, "model_config": model.config,
        "tokenizer_sha256": sha256(tok_path), "data_sha256": sha256(data_path),
        "heldout_sha256": sha256(heldout_path) if args.mode == "sft" else None,
        "hyperparameters": vars(args),
        "licenses": {"tokenizer": args.tokenizer_license,
            "data_sources": [{"source": r.get("source"), "license": r.get("license", "unverified")} for r in rows],
            "commercial_license_clearance": "not_established"},
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in
            (Path(__file__).resolve(), exp/"modeling"/"own_voice.py", root/"voz_distribuida.py",
             root/"models"/"biocognitive_projector.py", exp/"contracts"/"state.py",
             exp/"contracts"/"evidence.py", exp/"contracts"/"rendering.py")},
        "mode": args.mode, "condition": args.condition, "checkpoint_sha256": checkpoint_identity,
        "seed": args.seed, "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(),
        "commercial_readiness": "not_established"}
    atomic_json(run / "run_manifest.json", identity)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encoded(row):
        state, decision, prompt, bio, features = prepared[row["id"]]
        if build_response(row["response"], state, decision).response != row["response"].strip():
            raise ValueError("Supervised response contains forbidden channel boundaries")
        response_ids = tokenizer.encode(row["response"], add_special_tokens=False).ids
        if not response_ids or any(token in set(special.values()) for token in response_ids):
            raise ValueError("Response must contain nonempty ordinary tokens, without reserved controls")
        target = response_ids + [special["eos"]]
        inputs, targets = next_token_targets(prompt, target, model.max_len)
        x = torch.tensor([inputs], device="cuda", dtype=torch.long)
        labels = torch.tensor([targets], device="cuda", dtype=torch.long)
        return x, labels, bio, features

    if args.mode == "sft":
        batches = {r["id"]: encoded(r) for r in rows}
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1-step/args.max_steps))
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        cursor = 0
        for step in range(1, args.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.accumulation):
                row = train[cursor % len(train)]
                cursor += 1
                x, labels, bio, features = batches[row["id"]]
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(x, bio, features)
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss / args.accumulation).backward()
                losses.append(float(loss.detach()))
                throttle()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            record = {"step": step, "cursor": cursor, "train_loss": sum(losses)/len(losses), "grad_norm": float(norm)}
            if step % args.save_every == 0 or step == args.max_steps:
                model.eval()
                total, count = 0., 0
                with torch.inference_mode():
                    for row in validation:
                        x, labels, bio, features = batches[row["id"]]
                        with torch.autocast("cuda", dtype=dtype):
                            value = torch.nn.functional.cross_entropy(model(x, bio, features).flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
                        total += float(value)
                        count += int((labels != -100).sum())
                        throttle()
                record["validation_loss"] = total / count
                if not math.isfinite(record["validation_loss"]):
                    raise FloatingPointError("Nonfinite validation loss; checkpoint withheld")
                record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                checkpoint = run / f"step-{step:06d}.pt"
                fd, temporary = tempfile.mkstemp(dir=run, suffix=".tmp")
                os.close(fd)
                try:
                    torch.save({"architecture_version": model.architecture_version, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "cursor": cursor, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
                    os.replace(temporary, checkpoint)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                atomic_json(checkpoint.with_suffix(".json"), {**identity, "step": step,
                    "checkpoint_sha256": sha256(checkpoint), "metrics": record})
            with open(run/"metrics.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
    else:
        model.eval()
        with open(run/"responses.jsonl", "x", encoding="utf-8") as handle:
            for row in rows:
                state, decision, prompt, bio, features = prepared[row["id"]]
                if len(prompt) + args.max_new_tokens > model.max_len:
                    raise ValueError("Prompt plus output budget exceeds context; no silent truncation")
                if args.condition == "without_bio":
                    bio = torch.zeros_like(bio)  # diagnostic intervention only, not a physiological baseline
                if args.condition == "without_evidence":
                    features = None  # token-only ablation; symbolic evidence remains in prompt
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    generated = model.gerar(prompt, bio, features, max_new_tokens=args.max_new_tokens, temperatura=0.0)
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                envelope = build_response(raw, state, decision)
                result = {"id": row["id"], "response": envelope.response, "output_status": envelope.diagnostic.output_status,
                    "runtime": {"backend": "own_decoder", "mock": False, "artifact_sha256": checkpoint_identity},
                    "condition": args.condition, "ablation_scope": "cross_attention_token_only",
                    "prompt_tokens": len(prompt), "generated_tokens": len(generated),
                    "ended_eos": bool(generated and generated[-1] == special["eos"]),
                    "latency_s": time.perf_counter()-start}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                throttle()
    atomic_json(run/"RUN_COMPLETE.json", {"mode": args.mode, "commercial_readiness": "not_established",
        "manifest_sha256": sha256(run/"run_manifest.json"), "completed_at_unix": time.time()})


if __name__ == "__main__":
    main()
