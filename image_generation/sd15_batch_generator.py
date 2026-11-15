"""
SD 1.5 Bias-Prompt Batch Generator (Kaggle, T4/T4x2)
=====================================================

WHAT THIS DOES
- Reads your 420-row prompt xlsx from the Kaggle input dataset.
- Generates `IMAGES_PER_PROMPT` images per prompt into /kaggle/working (writable).
- Tracks progress in a manifest CSV, updated after every batch (not just every
  prompt), so a crash / disconnect / OOM only loses at most one batch's worth
  of images, not the whole prompt.
- Re-running this script (same cell, new session, whatever) automatically
  resumes from the manifest -- it will not regenerate finished work.
- Periodically pushes /kaggle/working as a NEW VERSION of your existing
  Kaggle dataset, so generated images persist beyond the notebook session.

ONE-TIME SETUP (do this before running)
1. Turn on internet for this notebook: Settings -> Internet -> On.
   (Needed only for the Kaggle API push step, not for the model itself.)
2. Add your Kaggle API credentials as Secrets:
   Add-ons -> Secrets -> add KAGGLE_USERNAME and KAGGLE_KEY
   (from kaggle.com/settings -> API -> Create New Token, open the
   downloaded kaggle.json to get these two values).
3. Set DATASET_SLUG below to your dataset's "owner/dataset-name" --
   from your path this is "ajfaisal002/t2i-images".
4. Double check INPUT_XLSX matches the actual filename in that dataset.

IMPORTANT LIMITATIONS TO KNOW ABOUT
- A push updates the dataset for FUTURE sessions, not the currently mounted
  read-only /kaggle/input copy in this same session.
- Kaggle: 12h max per session (GPU/CPU), ~30 GPU-hours/week quota. This script
  is built to be interrupted and re-run -- budget your run across sessions
  rather than expecting one continuous 30+ hour run.
- Run this once with a SMALL IMAGES_PER_PROMPT (e.g. 2) first to confirm the
  full pipeline (generation + checkpoint + dataset push) works end to end
  before committing your GPU quota to the full 420-prompt run.
- For 2x T4: launch this script twice (two processes / two notebook cells),
  once with GEN_DEVICE=cuda:0 GEN_SHARD=0/2 and once with
  GEN_DEVICE=cuda:1 GEN_SHARD=1/2, to split prompts across both GPUs.
  Manifests are kept per-shard to avoid write conflicts between the two.
"""

import os
import json
import math
import time
import subprocess
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline

# ---------------- CONFIG ----------------
INPUT_XLSX   = "/kaggle/input/datasets/ajfaisal002/t2i-images/ProfessionAdjective Bias.xlsx"
OUTPUT_ROOT  = "/kaggle/working/sd15_output"
DATASET_SLUG = "ajfaisal002/t2i-images"          # owner/dataset-name to push new versions to

IMAGES_PER_PROMPT   = 50     # drop to 2 for your first end-to-end test run
BATCH_SIZE          = 4      # try 4 first; raise to 8 if no OOM
STEPS               = 20
GUIDANCE            = 7.5
HEIGHT = WIDTH      = 512
PUSH_EVERY_N_PROMPTS = 5     # how often to checkpoint-push to the dataset

DEVICE = os.environ.get("GEN_DEVICE", "cuda:0")
SHARD  = os.environ.get("GEN_SHARD", "0/1")      # "shard_index/total_shards"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------- RESTORE STATE FROM PREVIOUS PUSH ----------------
# /kaggle/working resets to empty on a brand new session. Anything pushed in
# an earlier session only lives in the mounted input dataset now. Without
# this step, a session restart would (a) lose resume info and regenerate
# everything from scratch, and (b) the next push would overwrite the dataset
# with only this session's images, dropping earlier progress from the
# current version. This copies previously pushed manifests + images back in
# before the resume check runs.
import shutil

def restore_previous_state():
    src_root = os.path.dirname(INPUT_XLSX)
    if not os.path.isdir(src_root):
        return
    restored_any = False
    for name in os.listdir(src_root):
        if name == os.path.basename(INPUT_XLSX):
            continue  # skip the prompt xlsx itself
        src = os.path.join(src_root, name)
        dst = os.path.join(OUTPUT_ROOT, name)
        if os.path.exists(dst):
            continue  # already present locally this session -- don't clobber live progress
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        restored_any = True
    if restored_any:
        print("Restored previously pushed manifests/images from the input dataset into OUTPUT_ROOT.")
    else:
        print("No previous pushed state found (or already present locally) -- starting fresh.")

restore_previous_state()

# ---------------- LOAD PROMPTS ----------------
if not os.path.isfile(INPUT_XLSX):
    parent = os.path.dirname(INPUT_XLSX)
    actual = os.listdir(parent) if os.path.isdir(parent) else "(parent folder not found)"
    raise FileNotFoundError(
        f"INPUT_XLSX not found: {INPUT_XLSX}\n"
        f"Actual files in {parent}: {actual}\n"
        f"Update INPUT_XLSX above to match exactly."
    )

df = pd.read_excel(INPUT_XLSX)
df = df.reset_index().rename(columns={"index": "prompt_id"})

shard_idx, shard_total = (int(x) for x in SHARD.split("/"))
df = df[df["prompt_id"] % shard_total == shard_idx].reset_index(drop=True)
print(f"Shard {shard_idx}/{shard_total}: {len(df)} prompts assigned to {DEVICE}")

# ---------------- MANIFEST (per-shard to avoid write races) ----------------
MANIFEST_PATH = os.path.join(OUTPUT_ROOT, f"manifest_shard{shard_idx}.csv")

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame(columns=[
        "prompt_id", "profession", "category", "modifier", "prompt", "images_done"
    ])

manifest = load_manifest()
done_map = dict(zip(manifest.prompt_id, manifest.images_done)) if len(manifest) else {}

def save_manifest():
    manifest.to_csv(MANIFEST_PATH, index=False)

# ---------------- KAGGLE DATASET PUSH ----------------
def ensure_dataset_metadata():
    meta_path = os.path.join(OUTPUT_ROOT, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({
                "title": DATASET_SLUG.split("/")[-1],
                "id": DATASET_SLUG,
                "licenses": [{"name": "CC0-1.0"}]
            }, f)

def push_checkpoint(message="auto checkpoint"):
    ensure_dataset_metadata()
    try:
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", OUTPUT_ROOT,
             "-m", message, "-r", "zip"],
            check=True, capture_output=True, text=True
        )
        print(f"[checkpoint] pushed to {DATASET_SLUG}: {message}")
    except subprocess.CalledProcessError as e:
        # Non-fatal: generation continues even if a push fails (e.g. transient
        # network issue). Next cadence will retry.
        print(f"[checkpoint] push FAILED, will retry next cadence.\n{e.stderr}")

# ---------------- MODEL ----------------
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe.to(DEVICE)
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)
torch.cuda.empty_cache()

# ---------------- GENERATION LOOP ----------------
prompts_since_push = 0

with torch.inference_mode():
    for _, row in df.iterrows():
        pid = row.prompt_id
        already = int(done_map.get(pid, 0))
        if already >= IMAGES_PER_PROMPT:
            continue  # fully done -- resume skips it

        folder = os.path.join(
            OUTPUT_ROOT,
            str(row.profession).replace(" ", "_"),
            str(row.category),
            str(row.modifier),
            f"prompt_{pid}"
        )
        os.makedirs(folder, exist_ok=True)

        img_counter = already
        n_batches = math.ceil((IMAGES_PER_PROMPT - already) / BATCH_SIZE)
        print(f"[{pid}] {row.prompt}  ({already}/{IMAGES_PER_PROMPT} done)")

        for _b in range(n_batches):
            cur_bs = min(BATCH_SIZE, IMAGES_PER_PROMPT - img_counter)
            gens = [
                torch.Generator(DEVICE).manual_seed(1000 * pid + img_counter + j)
                for j in range(cur_bs)
            ]
            try:
                imgs = pipe(
                    [row.prompt] * cur_bs,
                    height=HEIGHT, width=WIDTH,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=STEPS,
                    generator=gens
                ).images
                for img in imgs:
                    img_counter += 1
                    img.save(f"{folder}/img_{img_counter}.png")
            except RuntimeError as e:
                print(f"[warn] batch failed on prompt {pid}: {e}")
                torch.cuda.empty_cache()
                time.sleep(5)
                continue  # this batch's images stay un-counted -> resumed later

            # checkpoint after every batch, not just every prompt
            if pid in manifest.prompt_id.values:
                manifest.loc[manifest.prompt_id == pid, "images_done"] = img_counter
            else:
                manifest.loc[len(manifest)] = [
                    pid, row.profession, row.category, row.modifier, row.prompt, img_counter
                ]
            save_manifest()

        prompts_since_push += 1
        if prompts_since_push >= PUSH_EVERY_N_PROMPTS:
            push_checkpoint(f"checkpoint through prompt {pid}")
            prompts_since_push = 0

push_checkpoint("final checkpoint for this run")
print("Run finished (or session ended). Re-run this script to resume automatically.")
