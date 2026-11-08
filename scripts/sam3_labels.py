#!/usr/bin/env python3
"""Per-pixel class labels for a splat's training images, with SAM 3.

SAM 3 (``facebook/sam3`` through Hugging Face transformers) finds every
instance of each text prompt in an image. Each pixel takes the prompt of the
highest-scoring instance that covers it, 0 ("other") where none does, and
255 (no label) where the image's alpha masks it out, as ``gsplat_photos``
leaves the pixels past ``--max-angle`` and under a ``--camera-mask``. Then
``gsplat_semantic_cuda`` votes the classes onto the gaussians:

    python scripts/sam3_labels.py --images run/images --out run/labels \\
        --prompts "chair,table,wall,floor,ceiling,door,window,screen"

Writes ``<out>/<image name>.png`` (the image's name with the extension
replaced; 8-bit grey, the class index) and ``<out>/classes.txt`` (``index r g
b name`` per line: 0 other, then the prompts in order). ``--preview`` also
writes ``<out>/preview/<image name>.jpg``, the classes over the photo.
``--prompt-threshold window=0.4`` keeps that prompt's instances from a
lower (or higher) score than ``--threshold``. ``--instances`` also writes
each pixel's instance, for
``gsplat_semantic_cuda --instances``: ``<out>/instances/<image name>.png``
(8-bit grey: 0 none, i the image's i-th instance, 255 no label) and ``.txt``
(``id class score`` per instance, best scores first).

Needs torch, a transformers with SAM 3 (``Sam3Model``), and access to the
gated ``facebook/sam3`` checkpoint on the Hugging Face Hub (``hf auth
login``).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# 20 colours that stay apart (Sasha Trubetskoy's list), for the prompts in
# order; "other" is grey.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (255, 255, 255),
]
OTHER = (128, 128, 128)
NO_LABEL = 255
MAX_INSTANCES = 254
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def image_names(root):
    """Image files under ``root``, as paths relative to it, sorted."""
    names = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                names.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(names)


def class_colors(num_prompts):
    return [OTHER] + [PALETTE[i % len(PALETTE)] for i in range(num_prompts)]


def encode_prompts(model, processor, prompts, device):
    """Each prompt's text features and attention mask, computed once."""
    out = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = processor(text=prompt, return_tensors="pt").to(device)
            out.append((model.get_text_features(**inputs), inputs.attention_mask))
    return out


def label_image(model, processor, rgb, texts, thresholds, mask_threshold, device, dtype):
    """Each pixel's class (0 other, k the k-th prompt) and instance (0 none,
    i the i-th entry of the returned list of (class, score), at most
    MAX_INSTANCES), both from the highest-scoring instance over it. The k-th
    prompt keeps instances scoring ``thresholds[k - 1]`` or more."""
    w, h = rgb.size
    label = np.zeros((h, w), np.uint8)
    inst = np.zeros((h, w), np.int32)
    best = np.zeros((h, w), np.float32)
    found = []  # (class, score) per instance, 1-based ids in `inst`
    image_inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        # The image's features once, then one detection per prompt.
        vision = model.get_vision_features(pixel_values=image_inputs.pixel_values.to(dtype))
        for k, ((text, mask), threshold) in enumerate(zip(texts, thresholds), start=1):
            outputs = model(vision_embeds=vision, text_embeds=text, attention_mask=mask)
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=image_inputs.get("original_sizes").tolist(),
            )[0]
            for mask, score in zip(result["masks"], result["scores"]):
                score = float(score)
                take = mask.cpu().numpy().astype(bool) & (score > best)
                found.append((k, score))
                label[take] = k
                inst[take] = len(found)
                best[take] = score
    # Renumber the instances that kept pixels, best scores first.
    kept = [i for i in np.unique(inst) if i > 0]
    kept.sort(key=lambda i: -found[i - 1][1])
    kept = kept[:MAX_INSTANCES]
    remap = np.zeros(len(found) + 1, np.uint8)
    for new, old in enumerate(kept, start=1):
        remap[old] = new
    return label, remap[inst], [found[i - 1] for i in kept]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--images", required=True, help="the images (e.g. <run>/images)")
    ap.add_argument("--out", required=True, help="the label directory to write")
    ap.add_argument("--prompts", required=True, help="comma-separated class prompts")
    ap.add_argument("--threshold", type=float, default=0.5, help="instance score to keep")
    ap.add_argument(
        "--prompt-threshold", action="append", default=[], metavar="PROMPT=SCORE",
        help="instance score to keep for one prompt, in place of --threshold "
        "(e.g. window=0.4); repeat for more prompts",
    )
    ap.add_argument("--mask-threshold", type=float, default=0.5, help="mask probability")
    ap.add_argument(
        "--rotate", type=int, default=0, choices=(0, 90, 180, 270),
        help="turn each image this many degrees counter-clockwise for SAM 3 "
        "(e.g. 180 for a camera mounted upside down); the labels are turned back",
    )
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--preview", action="store_true", help="also write <out>/preview/*.jpg")
    ap.add_argument(
        "--instances", action="store_true",
        help="also write <out>/instances/<image name>.png and .txt, each pixel's instance",
    )
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts or len(prompts) > 254:
        sys.exit("--prompts: 1 to 254 classes")
    thresholds = [args.threshold] * len(prompts)
    for spec in args.prompt_threshold:
        prompt, sep, score = spec.rpartition("=")
        if not sep or prompt.strip() not in prompts:
            sys.exit(f"--prompt-threshold {spec}: not PROMPT=SCORE with one of --prompts")
        thresholds[prompts.index(prompt.strip())] = float(score)
    names = image_names(args.images)
    if not names:
        sys.exit(f"no images in {args.images}")
    os.makedirs(args.out, exist_ok=True)
    colors = class_colors(len(prompts))
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("# index r g b name\n")
        for i, (name, c) in enumerate(zip(["other"] + prompts, colors)):
            f.write(f"{i} {c[0]} {c[1]} {c[2]} {name}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.model)
    texts = encode_prompts(model, processor, prompts, device)
    palette = np.array(colors + [(0, 0, 0)] * (256 - len(colors)), np.uint8)
    palette[NO_LABEL] = (0, 0, 0)

    counts = np.zeros(len(prompts) + 2, np.int64)  # classes, then no label
    t0 = time.time()
    for i, name in enumerate(names):
        img = Image.open(os.path.join(args.images, name))
        alpha = np.asarray(img.getchannel("A")) if "A" in img.getbands() else None
        rgb = img.convert("RGB")
        label, inst, found = label_image(
            model, processor, rgb.rotate(args.rotate, expand=True), texts,
            thresholds, args.mask_threshold, device, dtype,
        )
        label = np.ascontiguousarray(np.rot90(label, -args.rotate // 90))
        inst = np.ascontiguousarray(np.rot90(inst, -args.rotate // 90))
        if alpha is not None:
            label[alpha < 128] = NO_LABEL
            inst[alpha < 128] = NO_LABEL
        stem = os.path.splitext(name)[0]
        out = os.path.join(args.out, stem + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(label, "L").save(out)
        if args.instances:
            base = os.path.join(args.out, "instances", stem)
            os.makedirs(os.path.dirname(base), exist_ok=True)
            Image.fromarray(inst, "L").save(base + ".png")
            with open(base + ".txt", "w") as f:
                f.write("# id class score\n")
                for j, (k, score) in enumerate(found, start=1):
                    f.write(f"{j} {k} {score:.4f}\n")
        counts[: len(prompts) + 1] += np.bincount(
            label[label != NO_LABEL].ravel(), minlength=len(prompts) + 1
        )[: len(prompts) + 1]
        counts[-1] += int((label == NO_LABEL).sum())
        if args.preview:
            seen = np.asarray(rgb, np.float32)
            mixed = 0.5 * seen + 0.5 * palette[label].astype(np.float32)
            mixed[label == NO_LABEL] = seen[label == NO_LABEL] * 0.3
            prev = os.path.join(args.out, "preview", stem + ".jpg")
            os.makedirs(os.path.dirname(prev), exist_ok=True)
            Image.fromarray(mixed.astype(np.uint8)).save(prev, quality=90)
        if (i + 1) % 10 == 0 or i + 1 == len(names):
            print(f"{i + 1}/{len(names)} images, {(time.time() - t0) / (i + 1):.2f} s each", flush=True)

    labelled = counts[:-1].sum() or 1
    for name, c in zip(["other"] + prompts, counts[:-1]):
        print(f"  {100 * c / labelled:6.2f}% {name}")
    print(f"wrote {len(names)} label images and classes.txt to {args.out}")


if __name__ == "__main__":
    main()
