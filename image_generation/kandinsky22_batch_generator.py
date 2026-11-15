"""
Kandinsky 2.2 Batch Generator (Kaggle, T4/T4x2)
=================================================
Mirrors the SD1.5 / PixArt-Sigma batch generators: resilient, checkpointed,
resumes across both in-session crashes AND full session restarts, pushes to
the shared Kaggle dataset.

STRUCTURAL DIFFERENCE FROM THE OTHER TWO MODELS: Kandinsky 2.2 is a TWO-STAGE
pipeline -- a "prior" that turns the text prompt into image embeddings, and a
"decoder" that turns those embeddings into the actual image. Both need to be
loaded and both need to run per generation.

NOTE ON YOUR PILOT: your pilot code generated one image at a time (no
batching) and did NOT set a manual seed (unlike your SD1.5/PixArt pilots,
which did). This script preserves that -- unseeded, one image per prior+decoder
call -- to match exactly what you tested. If you want reproducible/seeded
generations for consistency with the other models, that's an easy addition,
just ask.

NOTE ON STEPS: your pilot *summary table* said steps=25, but the actual code
you gave me uses steps=30. This script uses 30 (the code, not the table) --
confirm that's the value you actually want for the full run.

SHARED OUTPUT ROOT: same as PixArt -- /kaggle/working/generated_images with
one subfolder per model, so pushes don't drop other models' data from the
dataset's current version.

ONE-TIME SETUP: same as the other two scripts -- internet on, KAGGLE_KEY /
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
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline

# ---------------- CONFIG ----------------
INPUT_XLSX   = "/kaggle/input/datasets/ajfaisal002/t2i-images/ProfessionAdjective Bias.xlsx"
OUTPUT_ROOT  = "/kaggle/working/generated_images"   # SHARED across all models
MODEL_NAME   = "kandinsky22"
MODEL_DIR    = os.path.join(OUTPUT_ROOT, MODEL_NAME)
DATASET_SLUG = "ajfaisal002/t2i-images"

IMAGES_PER_PROMPT    = 50     # drop to 2 for a first end-to-end test run
STEPS                = 30     # from your actual pilot code (table said 25 -- verify which you want)
GUIDANCE             = 4.0
HEIGHT = WIDTH       = 512
NEGATIVE_PROMPT      = "low quality, blurry"   # from your pilot
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

# ---------------- MODEL (two-stage: prior + decoder) ----------------
prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
)
prior.to(DEVICE)

pipe = KandinskyV22Pipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.to(DEVICE)
pipe.enable_attention_slicing()
torch.cuda.empty_cache()
print("Kandinsky 2.2 loaded successfully!")

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
        remaining = IMAGES_PER_PROMPT - already
        print(f"[{pid}] {row.prompt}  ({already}/{IMAGES_PER_PROMPT} done)")

        for _i in range(remaining):
            try:
                image_emb, negative_image_emb = prior(
                    prompt=row.prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=GUIDANCE
                ).to_tuple()

                image = pipe(
                    image_embeds=image_emb,
                    negative_image_embeds=negative_image_emb,
                    num_inference_steps=STEPS,
                    height=HEIGHT,
                    width=WIDTH
                ).images[0]

                img_counter += 1
                image.save(f"{folder}/img_{img_counter}.png")
            except RuntimeError as e:
                print(f"[warn] generation failed on prompt {pid}, image {img_counter+1}: {e}")
                torch.cuda.empty_cache()
                time.sleep(5)
                continue  # this image stays un-counted -> resumed later

            # checkpoint after every image (finer-grained than the batched models,
            # since Kandinsky here generates one image per call anyway)
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
