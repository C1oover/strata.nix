#!/usr/bin/env python3
"""Strata agnostic model prefetch & config generator for NixOS.

Supports:
  1. Presets (--family swift/qwen/coder/unsloth/gyro/qwen36/deepseek41 --model IQ2_XS)
  2. Any Hugging Face repo (--repo owner/repo [--revision main] [--file path.gguf] [--pattern '*IQ2_XS*.gguf'])
  3. Any direct URLs (--url https://...) or local files/directories (--gguf-dir /path or --model-path /path/model.gguf)
  4. Optional Vision (--mmproj-repo / --mmproj-file / --mmproj-path)
  5. Optional MTP / Speculative Drafter (--mtp auto|on|off, or --draft-repo / --draft-file / --draft-path)
  6. Optional DeepSeek Engram table (--engram-repo / --engram-file / --engram-path)
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SHARE = Path(os.environ.get("STRATA_SHARE", "")) if os.environ.get("STRATA_SHARE") else None
if SHARE is None:
    sys.exit("strata-prefetch: STRATA_SHARE is not set (run it through the strata package's wrapper)")

TOOLS = SHARE / "tools"
sys.path.insert(0, str(TOOLS))

HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Built-in presets remain available as defaults, but can be overridden by --repo / --file / --pattern
PRESETS = {
    "qwen": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "pattern": "{q}/*{q}*.gguf",
        "name": "qwen3.8-flash-next",
        "profile": "expert-profile.bin",
        "mode": "pack",
        "mtp": "on",
    },
    "swift": {
        "repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "pattern": "*{q}*.gguf",
        "name": "swift-1.5",
        "profile": "expert-profile.bin",
        "mode": "pack",
        "mtp": "on",
    },
    "abliterated": {
        "repo": "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
        "pattern": "{q}/*{q}*.gguf",
        "name": "swift-1.5-abliterated",
        "profile": "expert-profile.bin",
        "mode": "pack",
        "mtp": "on",
    },
    "coder": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        "pattern": "{q}/*{q}*.gguf",
        "name": "qwen3.8-flash-next-coder",
        "profile": "expert-profile-coder.bin",
        "mode": "pack",
        "mtp": "on",
    },
    "unsloth": {
        "repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "pattern": "*{q}*.gguf",
        "name": "qwen3.8-unsloth",
        "profile": "expert-profile.bin",
        "mode": "auto",
        "mtp": "on",
    },
    "gyro": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-Gyro-GGUF",
        "pattern": "*{q}*.gguf",
        "name": "qwen3.8-gyro",
        "profile": "expert-profile.bin",
        "mode": "pack",
        "mtp": "on",
    },
    "qwen36": {
        "repo": "Qwen/Qwen3.6-35B-A3B-GGUF",
        "pattern": "*{q}*.gguf",
        "name": "qwen3.6-35b-a3b",
        "profile": None,
        "mode": "gguf",
        "mtp": "off",
    },
    "deepseek41": {
        "repo": "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GGUF",
        "pattern": "*{q}*.gguf",
        "name": "deepseek-v4.1-flash",
        "profile": None,
        "mode": "auto",
        "mtp": "off",
    },
}

QUANT_RAM_GB = {
    "IQ1_S": 28,
    "IQ1_M": 32,
    "Q2_0": 48,
    "IQ2_XXS": 42,
    "IQ2_XS": 48,
    "IQ2_S": 52,
    "IQ3_XXS": 60,
    "IQ3_XS": 62,
    "IQ3_S": 62,
    "Q4_K_M": 48,
    "Q8_0": 64,
}


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"[strata-prefetch] ERROR: {msg}")
    if hint:
        say(f"  {hint}")
    sys.exit(1)


def ok(msg):
    say(f"[strata-prefetch] ok: {msg}")


def done(path: Path) -> bool:
    return path.with_name(path.name + ".done").exists()


def mark(path: Path):
    path.with_name(path.name + ".done").write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


def get_token(token_file: str | None) -> str | None:
    if token_file and Path(token_file).exists():
        tok = Path(token_file).read_text(encoding="utf-8").strip()
        if tok:
            return tok
    for env in HF_TOKEN_ENV:
        if os.environ.get(env):
            return os.environ[env].strip()
    return None


def token_headers(token: str | None = None) -> dict:
    headers = {"User-Agent": "strata-nix"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_list_repo_files(repo: str, revision: str = "main", token: str | None = None) -> list[str]:
    """Recursively list all files in a Hugging Face repo using the Hub API."""
    encoded_rev = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/models/{repo}/tree/{encoded_rev}?recursive=true"
    headers = token_headers(token)
    files: list[str] = []
    while url:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                items = json.loads(resp.read().decode("utf-8"))
                for item in items:
                    if item.get("type") == "file" and "path" in item:
                        files.append(item["path"])
                # Check Link header for pagination if huge repo
                link = resp.headers.get("Link", "")
                next_match = re.search(r'<([^>]+)>;\s*rel="next"', link)
                url = next_match.group(1) if next_match else ""
        except OSError as e:
            fail(f"failed to list files in Hugging Face repo '{repo}' (rev={revision}): {e}",
                 "check the repo name, network, or pass --hf-token-file if gated")
    return sorted(files)


def resolve_hf_files(repo: str, revision: str, files: list[str], patterns: list[str], quant: str | None,
                     token: str | None) -> list[str]:
    """Resolve explicit filenames or glob patterns against a Hugging Face repository."""
    if files and all("*" not in f and "?" not in f for f in files):
        return files

    all_repo_files = hf_list_repo_files(repo, revision, token)
    gguf_files = [f for f in all_repo_files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]

    search_patterns = list(files) + list(patterns)
    if not search_patterns:
        if quant:
            search_patterns = [f"*{quant}*.gguf", f"{quant}/*{quant}*.gguf", f"*{quant.lower()}*.gguf"]
        else:
            search_patterns = ["*.gguf"]

    matched: list[str] = []
    for pat in search_patterns:
        expanded = pat.format(q=quant or "") if "{q}" in pat else pat
        for rf in all_repo_files:
            if fnmatch.fnmatchCase(rf, expanded) or fnmatch.fnmatch(rf.lower(), expanded.lower()):
                if rf not in matched:
                    matched.append(rf)
            elif "/" not in expanded and fnmatch.fnmatch(Path(rf).name.lower(), expanded.lower()):
                if rf not in matched:
                    matched.append(rf)
        if matched:
            break

    if not matched and len(gguf_files) == 1:
        matched = gguf_files

    if not matched:
        sample = ", ".join(gguf_files[:10]) or "(no .gguf files found)"
        fail(f"no files in '{repo}' matched {search_patterns}",
             f"available GGUF files in {repo}: {sample}")

    return sorted(matched)


def hf_resolve_url(repo: str, revision: str, rpath: str) -> str:
    encoded_rev = urllib.parse.quote(revision, safe="")
    encoded_path = "/".join(urllib.parse.quote(p, safe="") for p in rpath.split("/"))
    return f"https://huggingface.co/{repo}/resolve/{encoded_rev}/{encoded_path}"


def download(url: str, dst: Path, what: str, token: str | None = None) -> None:
    """Resumable download with size check (.done mark short-circuits)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what} already present")
        return
    headers = token_headers(token)
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=headers)
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url} ({e})", "check network or URL and try again")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:
        mark(dst)
        ok(f"{what} already present")
        return
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req_headers = {**headers, "Range": f"bytes={have}-"} if have else headers
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 10:
                        last = time.time()
                        size = (f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)"
                                if total else f"{have / 1e6:7.1f} MB")
                        print(f"\r  {what}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            say(f"  download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}", "run again to resume")
    part.replace(dst)
    mark(dst)
    ok(f"{what} downloaded")


def check_shard(shard: Path) -> None:
    """Verify GGUF header and tensor bounds."""
    import gguf_reader as G
    if not shard.exists():
        fail(f"missing {shard}")
    try:
        g = G.GGUFFile(shard)
    except (ValueError, struct.error) as e:
        fail(f"{shard.name} is not a valid GGUF file ({e})", "delete it and run again")
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    if shard.stat().st_size < need:
        fail(f"{shard.name} is truncated: {shard.stat().st_size:,} of {need:,} bytes",
             "delete it and run again")


def find_tensor_shard(shards: list[Path], tensor_name: str) -> Path | None:
    import gguf_reader as G
    for s in shards:
        try:
            g = G.GGUFFile(s)
            if any(t.name == tensor_name for t in g.tensors):
                return s
        except Exception:
            continue
    return None


def run_tool(script: str, *args, env=None) -> None:
    cmd = [sys.executable, str(TOOLS / script), *(str(a) for a in args)]
    say("  > " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        fail(f"{script} failed (exit {r.returncode})")


def mem_total_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / 2**30
    except OSError:
        pass
    return 0.0


def slugify(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return s or hashlib.sha1(value.encode()).hexdigest()[:10]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Preset or Custom Repo Selection
    ap.add_argument("--family", default="swift",
                    help=f"Preset family ({', '.join(sorted(PRESETS))}) or 'custom'")
    ap.add_argument("--model", default="IQ2_XS",
                    help="Quantization tag (e.g. IQ2_XS, Q2_0, Q4_K_M) or model identifier")
    ap.add_argument("--model-name", help="Override the served model name in API responses")
    ap.add_argument("--repo", help="Any Hugging Face repo ID (e.g. 'owner/repo'), overrides --family preset")
    ap.add_argument("--revision", default="main", help="Hugging Face git revision/branch/commit (default: main)")
    ap.add_argument("--file", action="append", default=[],
                    help="Explicit file path(s) inside --repo (repeatable for multi-shard)")
    ap.add_argument("--pattern", action="append", default=[],
                    help="Glob pattern(s) to match files inside --repo (e.g. '*IQ2_XS*.gguf')")
    ap.add_argument("--url", action="append", default=[],
                    help="Direct URL(s) to download GGUF shards from (bypasses HF repo lookup)")
    ap.add_argument("--gguf-dir", help="Local directory containing already-downloaded GGUF file(s)")
    ap.add_argument("--model-path", action="append", default=[],
                    help="Direct local path(s) to existing GGUF file(s)")
    ap.add_argument("--pack-dir", help="Direct local path to an already built Strata pack directory")

    # Execution & Packing Mode
    ap.add_argument("--mode", choices=["auto", "pack", "gguf"], default="auto",
                    help="Engine loading mode: 'pack' (iq_pack.py + PLE), 'gguf' (direct GGUF), or 'auto' (inspect GGUF)")
    ap.add_argument("--expert-profile", help="Path or filename in SHARE/data for --expert-profile (use 'none' to disable)")
    ap.add_argument("--mtp", choices=["auto", "on", "off"], default="auto",
                    help="Fetch and enable MTP speculative draft layer (default: auto based on model)")
    ap.add_argument("--draft-repo", help="Optional HF repo for an external draft GGUF or custom MTP")
    ap.add_argument("--draft-file", help="Filename/glob in --draft-repo")
    ap.add_argument("--draft-path", help="Local path to a draft GGUF or MTP runtime directory")

    # Multimodal Vision (mmproj) & DeepSeek Engram support
    ap.add_argument("--mmproj-repo", help="Optional HF repo for multimodal vision encoder (mmproj GGUF)")
    ap.add_argument("--mmproj-file", help="Filename/glob in --mmproj-repo (default: '*mmproj*.gguf')")
    ap.add_argument("--mmproj-path", help="Local path to an existing mmproj GGUF file")
    ap.add_argument("--engram-repo", help="Optional HF repo for DeepSeek shared Engram table")
    ap.add_argument("--engram-file", help="Filename/glob in --engram-repo")
    ap.add_argument("--engram-path", help="Local path to an existing Engram file")

    # Runtime & Server Configuration
    ap.add_argument("--context", type=int, default=32768, help="Max context length in tokens (any positive integer)")
    ap.add_argument("--trained-context", type=int, default=262144,
                    help="Base trained context length before automatic YaRN scaling kicks in")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--low-ram", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--emit-config")
    ap.add_argument("--engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--gpu")
    ap.add_argument("--parallel", type=int, help="Number of parallel request slots")
    ap.add_argument("--lazy", action="store_true", help="Defer model loading in VRAM until first request")
    ap.add_argument("--api-key-file")
    ap.add_argument("--hf-token-file")
    ap.add_argument("--sampling-json")
    ap.add_argument("--mcp-json")
    ap.add_argument("--server-settings-json", help="Extra top-level keys to merge into emitted config.json")
    ap.add_argument("--extra-arg", action="append", default=[])
    a = ap.parse_args()

    data = Path(a.data_dir)
    models_dir = data / "models"
    token = get_token(a.hf_token_file)

    preset = PRESETS.get(a.family) if not a.repo and a.family in PRESETS else None
    repo = a.repo or (preset["repo"] if preset else None)
    mode = a.mode if a.mode != "auto" else (preset["mode"] if preset else "auto")
    mtp_mode = a.mtp if a.mtp != "auto" else (preset["mtp"] if preset else "auto")

    # Determine unique tag for pack & logs
    if repo:
        tag = slugify(f"{repo.split('/')[-1]}-{a.model}")
    elif a.model_path:
        tag = slugify(Path(a.model_path[0]).stem)
    elif a.gguf_dir:
        tag = slugify(f"{Path(a.gguf_dir).name}-{a.model}")
    else:
        tag = slugify(f"{a.family}-{a.model}")

    model_name = a.model_name or (preset["name"] if preset else tag)

    # 1. Resolve and acquire GGUF shards
    shards: list[Path] = []
    if a.model_path:
        shards = [Path(p) for p in a.model_path]
        ok(f"using explicit model paths: {[str(s) for s in shards]}")
    elif a.gguf_dir:
        gdir = Path(a.gguf_dir)
        pats = a.file or a.pattern or ([preset["pattern"].format(q=a.model)] if preset else [f"*{a.model}*.gguf", "*.gguf"])
        for pat in pats:
            expanded = pat.format(q=a.model) if "{q}" in pat else pat
            found = sorted(gdir.glob(expanded)) or sorted(gdir.rglob(Path(expanded).name))
            found = [f for f in found if f.is_file() and "mmproj" not in f.name.lower()]
            if found:
                shards = found
                break
        if not shards:
            fail(f"no GGUF files found in --gguf-dir {gdir} matching {pats}")
        ok(f"using local GGUF files from {gdir}: {[s.name for s in shards]}")
    elif a.url:
        for idx, u in enumerate(a.url, 1):
            fname = Path(urllib.parse.urlparse(u).path).name or f"{tag}-shard-{idx}.gguf"
            dst = models_dir / tag / fname
            download(u, dst, f"{tag} ({idx}/{len(a.url)})", token=token)
            shards.append(dst)
    elif repo:
        pats = a.pattern or ([preset["pattern"]] if preset and not a.file else [])
        rfiles = resolve_hf_files(repo, a.revision, a.file, pats, a.model, token)
        for idx, rf in enumerate(rfiles, 1):
            url = hf_resolve_url(repo, a.revision, rf)
            dst = models_dir / slugify(repo) / rf
            download(url, dst, f"{repo}/{rf} ({idx}/{len(rfiles)})", token=token)
            shards.append(dst)
    elif not a.pack_dir:
        fail("no model source specified: set --family, --repo, --url, --gguf-dir, or --model-path")

    for s in shards:
        check_shard(s)

    shard1 = shards[0] if shards else None
    ple = find_tensor_shard(shards, "per_layer_token_embd.weight") if shards else None

    if mode == "auto":
        mode = "pack" if (ple is not None or a.pack_dir) else "gguf"

    # Estimate RAM threshold for auto low-ram
    ram_threshold = QUANT_RAM_GB.get(a.model, 48)
    low_ram = a.low_ram
    if low_ram == "auto":
        low_ram = "on" if mem_total_gb() < ram_threshold else "off"

    args: list[str] = []
    tokenizer_path: str | None = None

    # 2. Build or configure Pack vs direct GGUF mode
    if mode == "pack":
        if shard1 is None and not a.pack_dir:
            fail("pack mode requires GGUF shards or --pack-dir")
        pack = Path(a.pack_dir) if a.pack_dir else (data / "packs" / tag)
        if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
            say(f"[strata-prefetch] building pack in {pack} (iq_pack.py) ...")
            run_tool("iq_pack.py", "--gguf", shard1, "--out", pack)
            ok(f"pack built: {pack}")
        else:
            ok(f"pack already built: {pack}")

        if low_ram == "on":
            if not (pack / "experts.bin").exists():
                say("[strata-prefetch] building experts.bin (low-RAM mmap mode) ...")
                run_tool("iq_pack.py", "--gguf", shard1, "--out", pack, "--experts-bin")
            ok("low-RAM mode: mapping experts from experts.bin")

        args += ["--pack", str(pack)]
        if shard1:
            args += ["--native", str(shard1)]
        if ple:
            args += ["--ple-gguf", str(ple)]

        # Expert profile resolution
        prof_arg = a.expert_profile if a.expert_profile is not None else (preset.get("profile") if preset else "expert-profile.bin")
        if prof_arg and prof_arg.lower() != "none":
            prof_path = Path(prof_arg) if "/" in prof_arg else (SHARE / "data" / prof_arg)
            if prof_path.exists():
                args += ["--expert-profile", str(prof_path)]

        args += ["--expert-cache", "auto", "--prefill", "auto"]
        if low_ram == "on":
            args += ["--mmap-experts"]

        tokenizer_path = str(pack / "tokenizer")
    else:
        # Direct GGUF mode (e.g. Qwen3.6-35B-A3B, DeepSeek V4.1, or standard GGUF repos)
        if not shard1:
            fail("gguf mode requires at least one GGUF file")
        args += ["--model", str(shard1)]
        if ple and ple != shard1:
            args += ["--ple-gguf", str(ple)]

    # 3. MTP / Speculative Drafter handling
    if mtp_mode == "auto":
        mtp_mode = "on" if mode == "pack" else "off"

    if a.draft_path:
        args += ["--spec", "4", "--spec-min-p", "0.5", "--mtp", str(a.draft_path)]
    elif a.draft_repo:
        dfiles = resolve_hf_files(a.draft_repo, "main", [a.draft_file] if a.draft_file else [], ["*.gguf"], None, token)
        ddst = models_dir / slugify(a.draft_repo) / dfiles[0]
        download(hf_resolve_url(a.draft_repo, "main", dfiles[0]), ddst, f"draft {a.draft_repo}/{dfiles[0]}", token=token)
        check_shard(ddst)
        args += ["--spec", "4", "--spec-min-p", "0.5", "--mtp", str(ddst)]
    elif mtp_mode == "on":
        mtp = data / "mtp"
        rt = mtp / "rt"
        if not (rt / "experts.bin").exists():
            say("[strata-prefetch] fetching and packing MTP draft layer (~5 GB, once) ...")
            run_tool("mtp_fetch.py", "fetch", "--out", mtp)
            run_tool("mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf")
            run_tool("mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", rt)
        if (SHARE / "data" / "draft_vocab.bin").exists() and not (rt / "draft_vocab.bin").exists():
            shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
        ok(f"MTP ready: {rt}")
        args += ["--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt)]

    # 4. Optional DeepSeek Engram table
    engram_path: Path | None = None
    if a.engram_path:
        engram_path = Path(a.engram_path)
    elif a.engram_repo:
        efiles = resolve_hf_files(a.engram_repo, "main", [a.engram_file] if a.engram_file else [], ["*.gguf", "*.bin"], None, token)
        engram_path = models_dir / slugify(a.engram_repo) / efiles[0]
        download(hf_resolve_url(a.engram_repo, "main", efiles[0]), engram_path, f"engram {a.engram_repo}/{efiles[0]}", token=token)
    if engram_path and not any(x == "--engram" for x in a.extra_arg):
        args += ["--engram", str(engram_path)]

    # 5. Optional Vision Encoder (mmproj)
    mmproj_path: Path | None = None
    if a.mmproj_path:
        mmproj_path = Path(a.mmproj_path)
    elif a.mmproj_repo:
        mfiles = resolve_hf_files(
            a.mmproj_repo,
            "main",
            [a.mmproj_file] if a.mmproj_file else [],
            ["*mmproj*.gguf", "*.gguf"],
            None,
            token,
        )
        mmproj_path = models_dir / slugify(a.mmproj_repo) / mfiles[0]
        download(hf_resolve_url(a.mmproj_repo, "main", mfiles[0]), mmproj_path, f"mmproj {a.mmproj_repo}/{mfiles[0]}", token=token)
        check_shard(mmproj_path)

    # 6. Context & KV Cache Streaming flags
    args += ["--max-context", str(a.context)]
    if a.context >= 65536 and low_ram != "on" and not any(x == "--kv-resident" for x in a.extra_arg):
        args += ["--kv-resident", "32768"]
    if a.context > a.trained_context and not any(x == "--rope-scaling" for x in a.extra_arg):
        args += ["--rope-scaling", "yarn", "--rope-scale", f"{a.context / a.trained_context:g}"]

    args += a.extra_arg

    prepared = {
        "args": args,
        "model_name": model_name,
        "context": a.context,
        "low_ram": low_ram == "on",
        "mode": mode,
        "shards": [str(s) for s in shards],
    }
    if tokenizer_path:
        prepared["tokenizer"] = tokenizer_path
    if mmproj_path:
        prepared["vision"] = {"mmproj": str(mmproj_path)}

    (data / f"prepared-{tag}.json").write_text(json.dumps(prepared, indent=1), encoding="utf-8")
    ok(f"prepared ({mode}): model={model_name} context={a.context}")

    if a.emit_config:
        if not a.engine:
            fail("--emit-config needs --engine")
        cfg: dict = {
            "exe": a.engine,
            "args": args,
            "model_name": model_name,
            "log": str(data / f"engine-{tag}.log"),
        }
        if tokenizer_path:
            cfg["tokenizer"] = tokenizer_path
        if mmproj_path:
            cfg["vision"] = {"mmproj": str(mmproj_path)}
        if a.host:
            cfg["host"] = a.host
        if a.port:
            cfg["port"] = a.port
        if a.gpu:
            cfg["gpu"] = a.gpu
        if a.parallel:
            cfg["parallel"] = a.parallel
        if a.lazy:
            cfg["lazy_load"] = True
        if a.api_key_file and Path(a.api_key_file).exists():
            key_val = Path(a.api_key_file).read_text(encoding="utf-8").strip()
            # Upstream server.py aborts if api_key is set to an empty string ""
            if key_val:
                cfg["api_key"] = key_val
        if a.sampling_json:
            cfg["sampling"] = json.loads(Path(a.sampling_json).read_text(encoding="utf-8"))
        if a.mcp_json:
            cfg["mcp_servers"] = json.loads(Path(a.mcp_json).read_text(encoding="utf-8"))
        if a.server_settings_json:
            extra_srv = json.loads(Path(a.server_settings_json).read_text(encoding="utf-8"))
            cfg.update(extra_srv)

        out = Path(a.emit_config)
        out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if "api_key" in cfg:
            os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
        ok(f"config written: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
