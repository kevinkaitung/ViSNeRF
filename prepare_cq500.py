#!/usr/bin/env python3
"""
Adapt the rendered CQ500 dataset into the schema ViSNeRF's `ViSNeRFDataset`
(data_loader.py) expects, and print the matching config block.

Input layout (one dir per scene, scene = <patient>_<tfidx>):

    <root>/CQ500CT0_000/gaussian_splat/transforms.json
    <root>/CQ500CT0_000/gaussian_splat/images/view_0.jpg ... view_49.jpg
    <root>/CQ500CT0_001/...

Each scene's transforms.json is the nerfstudio-style file written by
exps/produce_CQ500_images.py: top-level fl_x/fl_y/cx/cy/w/h and a `frames`
list of {file_path, transform_matrix} where transform_matrix is c2w in the
OpenGL convention. transforms.json is identical across a patient's TFs and
differs between patients (camera distance tracks the volume's x-extent).

What this script produces (nothing is re-rendered, images are referenced by
absolute path):

  transforms_train.json / transforms_test.json  in --out-dir, each with
    camera_angle_x       : from fl_x  (2*atan(0.5*w/fl_x))
    pose_normalization   : {scene: {center, scale}} so every scene lands in the
                           shared bbox -- center = least-squares look-at of that
                           scene's cameras, scale = target_radius / median(radius)
    frames[i]            : {file_path (abs), transform_matrix (raw), params, scene}

  cq500*.txt            a ready-to-run ViSNeRF config

Split (per the baseline decision):
  * TFs      -- whole transfer functions are held out. Endpoints (first/last TF
                index) always stay in train so the test TFs are interpolation
                targets, never extrapolation. Controlled by --tf-holdout.
  * Views    -- --n-train-views random views/scene go to train; the rest are
                "held-out views". Test = held-out views x held-out TFs by
                default (novel view AND novel TF, ViSNeRF-paper style); use
                --test-view-mode all to test held-out TFs at every view.
  * --no-holdout : no split at all. Every TF and every view goes into train,
                and transforms_test.json is an identical copy of the train set.
                Reports the train-set reconstruction PSNR (an upper bound; no
                generalization is measured). Ignores --tf-holdout /
                --n-train-views / --test-view-mode.

Layouts:
  combined     one train/test pair; params = [tf] or [patient, tf]     (K=1 or 2)
  per-patient  a subdir per patient, each its own train/test + config; params=[tf]

Examples:
  ./prepare_cq500.py --root .../CQ500_processed_new
  ./prepare_cq500.py --root R --param-axes patient,tf --tf-holdout stride:2
  ./prepare_cq500.py --root R --layout per-patient --n-train-views 45
"""
import argparse
import json
import math
import os
import re
import sys
import random
from collections import OrderedDict

import numpy as np

SCENE_RE = re.compile(r"^(?P<patient>.+?)_(?P<tf>\d+)$")


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def discover(root):
    """{patient: OrderedDict{tf_suffix: scene_dir_abspath}}, tf order = sorted."""
    if not os.path.isdir(root):
        sys.exit(f"root not found: {root}")
    scenes = {}
    unparsed = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        m = SCENE_RE.match(entry.name)
        tj = os.path.join(entry.path, "gaussian_splat", "transforms.json")
        if not m or not os.path.isfile(tj):
            unparsed.append(entry.name)
            continue
        scenes.setdefault(m.group("patient"), {})[m.group("tf")] = entry.path
    if unparsed:
        print(f"  note: ignored {len(unparsed)} dir(s) without <patient>_<tf>/gaussian_splat/"
              f"transforms.json: {', '.join(unparsed[:5])}"
              + (" ..." if len(unparsed) > 5 else ""))
    if not scenes:
        sys.exit("no usable scenes under root")
    for p in scenes:
        scenes[p] = OrderedDict(sorted(scenes[p].items()))
    return OrderedDict(sorted(scenes.items()))


def load_transforms(scene_dir):
    with open(os.path.join(scene_dir, "gaussian_splat", "transforms.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def camera_angle_x(meta):
    return 2.0 * math.atan(0.5 * meta["w"] / meta["fl_x"])


def lookat_and_radius(frames):
    """Least-squares point closest to every camera's optical axis, + radii.

    transform_matrix is c2w OpenGL: translation = col 3, forward = -col 2.
    """
    T = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    o = T[:, :3, 3]
    d = -T[:, :3, 2]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for oi, di in zip(o, d):
        P = np.eye(3) - np.outer(di, di)
        A += P
        b += P @ oi
    center = np.linalg.solve(A, b)
    radii = np.linalg.norm(o - center, axis=1)
    return center, radii


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def split_tfs(n_tf, spec):
    """Return (train_idx, test_idx) global TF indices. Endpoints stay in train."""
    if n_tf == 1:
        return [0], []
    kind, _, val = spec.partition(":")
    allidx = list(range(n_tf))
    if kind == "stride":
        s = max(2, int(val))
        train = sorted(set(allidx[::s]) | {0, n_tf - 1})
    elif kind == "frac":
        f = float(val)
        n_test = max(1, min(n_tf - 2, round(n_tf * f)))
        interior = allidx[1:-1]
        step = len(interior) / n_test
        test = {interior[min(len(interior) - 1, int((k + 0.5) * step))] for k in range(n_test)}
        train = [i for i in allidx if i not in test]
    elif kind == "indices":
        test = {int(x) for x in val.split(",") if x != ""}
        if 0 in test or (n_tf - 1) in test:
            sys.exit("--tf-holdout indices: cannot hold out the first or last TF (need endpoints in train)")
        train = [i for i in allidx if i not in test]
    else:
        sys.exit(f"--tf-holdout: unknown spec {spec!r} (use stride:K | frac:F | indices:a,b,c)")
    test = [i for i in allidx if i not in set(train)]
    if not test:
        sys.exit(f"--tf-holdout {spec!r} left no test TFs (n_tf={n_tf})")
    return train, test


def split_views(n_views, n_train, rng):
    idx = list(range(n_views))
    rng.shuffle(idx)
    n_train = min(n_train, n_views - 1) if n_views > 1 else n_views
    return sorted(idx[:n_train]), sorted(idx[n_train:])


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #
def build_frames(scene_dir, scene_name, meta, view_idx, params):
    out = []
    for vi in view_idx:
        fr = meta["frames"][vi]
        path = os.path.abspath(os.path.join(scene_dir, "gaussian_splat", fr["file_path"]))
        if not os.path.isfile(path):
            sys.exit(f"missing image: {path}")
        out.append({
            "file_path": path,
            "transform_matrix": fr["transform_matrix"],
            "params": list(params),
            "scene": scene_name,
        })
    return out


CONFIG_TEMPLATE = """\
# generated by prepare_cq500.py  --  {cmd}
dataset_name = visnerf
dataset = cq500
datadir = {datadir}
expname = {expname}
basedir = ./log

input_res = [256,256]
output_res = [256,256]

# --- parameter axes: {axes_desc} ---
nParams = {n_params}
n_lamb_params = {n_lamb_params}
vecSize_params = {vec_size}     # = #distinct TRAIN values per axis
min_params = {min_params}
max_params = {max_params}       # full index range so held-out TFs interpolate

# --- scene fit (poses are canonicalized to radius {target_radius} in the loader) ---
bbox = {bbox}
near_far = [{near}, {far}]
# black background: leave white_bkgd unset

n_iters = {n_iters}
batch_size = 4096

N_voxel_init = {n_voxel_init}
N_voxel_final = {n_voxel_final}
upsamp_list = [6000,9000,12000,16500,21000]
update_AlphaMask_list = [6000,12000]
use_AlphaMask = 0         # set 1 (paper) once geometry is verified

N_vis = 3
vis_every = {vis_every}
render_test = 1
render_path = 0

n_lamb_sigma = [16,16,16]
n_lamb_sh = [48,48,48]
model_name = ViSNeRF

shadingMode = MLP_Fea
fea2denseAct = softplus
view_pe = 2
fea_pe = 2

L1_weight_inital = 8e-5
L1_weight_rest = 4e-5
rm_weight_mask_thre = 1e-4
TV_weight_density = 0
TV_weight_app = 1.0
"""


def write_json(path, camera_angle, pose_norm, frames):
    with open(path, "w") as f:
        json.dump({
            "camera_angle_x": camera_angle,
            "pose_normalization": pose_norm,
            "frames": frames,
        }, f, indent=1)
    print(f"  wrote {path}  ({len(frames)} frames)")


def emit(out_dir, tag, datadir, camera_angle, pose_norm, train_frames, test_frames,
         axes, n_tf_total, n_patients, vec_size, target_radius, bbox_h, near, far,
         n_iters, quick, cmd):
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "transforms_train.json"), camera_angle, pose_norm, train_frames)
    write_json(os.path.join(out_dir, "transforms_test.json"), camera_angle, pose_norm, test_frames)

    if axes == ["tf"]:
        n_params, min_p, max_p, nlp = 1, [0], [n_tf_total - 1], [4]
    else:
        n_params, min_p, max_p, nlp = 2, [0, 0], [n_patients - 1, n_tf_total - 1], [4, 4]

    cfg = CONFIG_TEMPLATE.format(
        cmd=cmd,
        datadir=datadir,
        expname=f"cq500_{tag}" + ("_quick" if quick else ""),
        axes_desc=", ".join(axes),
        n_params=n_params,
        n_lamb_params=_l(nlp),
        vec_size=_l(vec_size),
        min_params=_l(min_p),
        max_params=_l(max_p),
        target_radius=target_radius,
        bbox=f"[[{-bbox_h},{-bbox_h},{-bbox_h}],[{bbox_h},{bbox_h},{bbox_h}]]",
        near=near, far=far,
        n_iters=6000 if quick else n_iters,
        n_voxel_init="2097152    # 128**3",
        n_voxel_final=("2097152  # 128**3 (quick: no upsampling)" if quick
                       else "27000000  # 300**3"),
        vis_every=2000 if quick else 5000,
    )
    tag = tag + ("_quick" if quick else "")
    cfg_path = os.path.join("configs", f"cq500_{tag}.txt")
    os.makedirs("configs", exist_ok=True)
    with open(cfg_path, "w") as f:
        f.write(cfg)
    print(f"  wrote {cfg_path}")
    print("\n" + "=" * 70 + f"\nconfig  configs/cq500_{tag}.txt\n" + "=" * 70)
    print(cfg)


def _l(xs):
    return "[" + ",".join(str(x) for x in xs) + "]"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir holding <patient>_<tf>/ scene dirs")
    ap.add_argument("--out-dir", default=None, help="where transforms_*.json go (default: --root)")
    ap.add_argument("--param-axes", default="tf", choices=("tf", "patient,tf"),
                    help="parameter axes to write into `params` (default: tf)")
    ap.add_argument("--layout", default="combined", choices=("combined", "per-patient"),
                    help="one model over all scenes, or a subdir/config per patient")
    ap.add_argument("--patients", default=None, help="comma-separated subset of patient ids")
    ap.add_argument("--tf-holdout", default="stride:2",
                    help="stride:K | frac:F | indices:a,b,c   (default stride:2)")
    ap.add_argument("--n-train-views", type=int, default=45, help="train views per scene (rest held out)")
    ap.add_argument("--test-view-mode", default="heldout", choices=("heldout", "all"),
                    help="test held-out TFs at held-out views only, or at every view")
    ap.add_argument("--no-holdout", action="store_true",
                    help="no split: all TFs + all views in train, test = identical copy "
                         "(train-set reconstruction PSNR). Ignores the three flags above.")
    ap.add_argument("--view-seed", type=int, default=0)
    ap.add_argument("--target-radius", type=float, default=3.0, help="canonical camera radius")
    ap.add_argument("--bbox-halfside", type=float, default=1.5)
    ap.add_argument("--near", type=float, default=1.0)
    ap.add_argument("--far", type=float, default=5.0)
    ap.add_argument("--n-iters", type=int, default=90000)
    ap.add_argument("--quick", action="store_true",
                    help="emit a fast sanity config (6k iters, 128^3 grid, vis every 2k) "
                         "to verify geometry before a full run")
    args = ap.parse_args()

    cmd = "prepare_cq500.py " + " ".join(sys.argv[1:])
    axes = args.param_axes.split(",")
    if args.layout == "per-patient" and axes != ["tf"]:
        sys.exit("--layout per-patient implies a single patient per model; use --param-axes tf")

    scenes = discover(args.root)
    if args.patients:
        keep = set(args.patients.split(","))
        scenes = OrderedDict((p, v) for p, v in scenes.items() if p in keep)
        if not scenes:
            sys.exit("--patients left no scenes")
    patients = list(scenes)
    out_root = os.path.abspath(args.out_dir or args.root)

    # camera_angle_x + pose_normalization: from each scene's own transforms.json
    metas, pose_norm, cax = {}, {}, None
    for p in patients:
        for tf, sdir in scenes[p].items():
            name = f"{p}_{tf}"
            m = load_transforms(sdir)
            metas[name] = m
            a = camera_angle_x(m)
            cax = a if cax is None else cax
            if abs(a - cax) > 1e-6:
                sys.exit(f"{name}: camera_angle_x {a:.5f} != {cax:.5f} (mixed FOV unsupported)")
            center, radii = lookat_and_radius(m["frames"])
            pose_norm[name] = {
                "center": [round(float(c), 6) for c in center],
                "scale": float(args.target_radius / np.median(radii)),
            }
    n_views = len(metas[f"{patients[0]}_{list(scenes[patients[0]])[0]}"]["frames"])
    rng = random.Random(args.view_seed)
    if args.no_holdout:
        tr_views, te_views = list(range(n_views)), []
    else:
        # one shared view split: the same viewpoints are train / held-out for
        # every scene, so the test set is "the same held-out views, novel TF".
        tr_views, te_views = split_views(n_views, args.n_train_views, rng)

    print("=" * 70)
    print(f"  patients        : {len(patients)}  ({', '.join(patients)})")
    if args.no_holdout:
        print(f"  views / scene   : {n_views}  -> ALL in train (--no-holdout)")
    else:
        print(f"  views / scene   : {n_views}  -> train {len(tr_views)}, held out {te_views}")
    print(f"  camera_angle_x  : {cax:.5f} rad ({math.degrees(cax):.1f} deg)")
    print(f"  param axes      : {axes}   layout: {args.layout}")

    def make_side(patient_subset, tag, out_dir, datadir):
        # TFs must match across the patients that share this model's grid
        tf_list = list(scenes[patient_subset[0]])
        for p in patient_subset[1:]:
            if list(scenes[p]) != tf_list:
                sys.exit(f"patient {p} TFs {list(scenes[p])} != {patient_subset[0]}'s "
                         f"{tf_list}; cannot share a TF axis")
        n_tf = len(tf_list)
        if args.no_holdout:
            train_tf, test_tf = list(range(n_tf)), []
            print(f"  [{tag}] TFs {n_tf} -> ALL in train (--no-holdout)")
        else:
            train_tf, test_tf = split_tfs(n_tf, args.tf_holdout)
            print(f"  [{tag}] TFs {n_tf} -> train {train_tf}  test {test_tf}")

        train_frames, test_frames = [], []
        for pi, p in enumerate(patient_subset):
            for gi, tf in enumerate(tf_list):
                name = f"{p}_{tf}"
                sdir = scenes[p][tf]
                params = [pi, gi] if axes == ["patient", "tf"] else [gi]
                if gi in train_tf:
                    train_frames += build_frames(sdir, name, metas[name], tr_views, params)
                if gi in test_tf:
                    te = list(range(n_views)) if args.test_view_mode == "all" else te_views
                    test_frames += build_frames(sdir, name, metas[name], te, params)
        if args.no_holdout:
            test_frames = list(train_frames)   # eval on the exact training set
        subset = set(patient_subset)
        pn = {k: v for k, v in pose_norm.items()
              if SCENE_RE.match(k).group("patient") in subset}
        vec = [len(train_tf)] if axes == ["tf"] else [len(patient_subset), len(train_tf)]
        emit(out_dir, tag, datadir, cax, pn, train_frames, test_frames,
             axes, n_tf, len(patient_subset), vec,
             args.target_radius, args.bbox_halfside, args.near, args.far,
             args.n_iters, args.quick, cmd)

    if args.layout == "combined":
        make_side(patients, "all" if len(patients) > 1 else patients[0], out_root, out_root)
    else:
        for p in patients:
            d = os.path.join(out_root, p)
            make_side([p], p, d, d)


if __name__ == "__main__":
    main()
