"""
PixArt-Sigma Batch Generator (Kaggle, T4/T4x2)
================================================
Mirrors the SD1.5 batch generator: resilient, checkpointed, resumes across
both in-session crashes AND full session restarts, pushes to the shared
Kaggle dataset.

KEY DIFFERENCE FROM THE SD1.5 SCRIPT: this uses a SHARED OUTPUT_ROOT
(/kaggle/working/generated_images) with one subfolder per model, instead of
a model-specific root.

WHY: `kaggle datasets version -p <path>` uploads whatever is in <path> as the
ENTIRE new dataset version -- it does not merge with other models' folders.
If SD1.5 and PixArt each pushed their own separate root, whichever model ran
last would push a version containing ONLY that model's images, silently
dropping the other model's data from the dataset's current version (still
recoverable from version history, but not what you want for the detection
stage that needs all 5 models together).

>>> ACTION NEEDED: update your SD1.5 script's OUTPUT_ROOT to
    "/kaggle/working/generated_images" (with images going into a
    "sd15" subfolder) to match this one, before running both models.
    I can do that edit for you if you'd like -- just ask.

ONE-TIME SETUP: same as the SD1.5 script -- internet on, KAGGLE_KEY /
KAGGLE_USERNAME secrets mapped into os.environ, DATASET_SLUG matching your
dataset's owner/name.
"""

import os
import json
import math
import time
import subprocess
import shutil
import pandas as pd
import torch
from diffusers import PixArtSigmaPipeline

# ---------------- CONFIG ----------------
INPUT_XLSX   = "/kaggle/input/datasets/ajfaisal002/t2i-images/ProfessionAdjective Bias.xlsx"
OUTPUT_ROOT  = "/kaggle/working/generated_images"   # SHARED across all models
MODEL_NAME   = "pixart_sigma"
MODEL_DIR    = os.path.join(OUTPUT_ROOT, MODEL_NAME)
DATASET_SLUG = "ajfaisal002/t2i-images"

IMAGES_PER_PROMPT    = 50     # drop to 2 for a first end-to-end test run
BATCH_SIZE           = 2      # matches your pilot; raise if no OOM
STEPS                = 6
GUIDANCE             = 4.0
HEIGHT = WIDTH       = 512    # kept as your tested pilot config, even though
                               # this checkpoint's native res is 1024 -- your
                               # pilot rated quality "Good" at 512, so left as-is
PUSH_EVERY_N_PROMPTS = 5

DEVICE = os.environ.get("GEN_DEVICE", "cuda:0")
SHARD  = os.environ.get("GEN_SHARD", "0/1")     # "shard_index/total_shards"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- RESTORE STATE FROM PREVIOUS PUSH ----------------
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
        print("Restored previously pushed manifests/images (all models) into OUTPUT_ROOT.")
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

# ---------------- MANIFEST (per model, per-shard) ----------------
MANIFEST_PATH = os.path.join(OUTPUT_ROOT, f"manifest_{MODEL_NAME}_shard{shard_idx}.csv")

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
        print(f"[checkpoint] push FAILED, will retry next cadence.\n{e.stderr}")

# ---------------- MODEL ----------------
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload(device=DEVICE)   # kept from your pilot (best for P100/T4);
                                                 # device= added so 2-GPU sharding offloads
                                                 # onto the correct GPU per process
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
            MODEL_DIR,
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
            generators = [
                torch.Generator(DEVICE).manual_seed(1000 * pid + img_counter + j)
                for j in range(cur_bs)
            ]
            try:
                images = pipe(
                    [row.prompt] * cur_bs,
                    height=HEIGHT, width=WIDTH,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=STEPS,
                    clean_caption=False,
                    generator=generators
                ).images
                for img in images:
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
            push_checkpoint(f"[{MODEL_NAME}] checkpoint through prompt {pid}")
            prompts_since_push = 0

push_checkpoint(f"[{MODEL_NAME}] final checkpoint for this run")
print("Run finished (or session ended). Re-run this script to resume automatically.")
