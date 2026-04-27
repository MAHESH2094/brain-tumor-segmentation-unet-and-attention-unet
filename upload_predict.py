# ===================================================
# CELL 13: Upload -> Predict Runner (FINAL STEP)
# ===================================================
# Purpose: Final user-facing step after all pipeline cells.
#          - Load trained model
#          - Discover uploaded patient folders
#          - Run single or batch inference
#          - Print and optionally save summary JSON

import json
import os
import importlib.util
import numpy as np
import time
import itertools

try:
    from PIL import Image
except Exception:
    Image = None


def _infer_modality_from_filename(filename):
    """Map filename to expected modality tag, or None when unknown."""
    name = filename.lower()
    if not (name.endswith('.nii') or name.endswith('.nii.gz')):
        return None

    if 'flair' in name:
        return 'flair'
    if 't1ce' in name or 't1-ce' in name or 't1_ce' in name:
        return 't1ce'
    if 't2' in name:
        return 't2'
    if 't1' in name:
        return 't1'
    return None


def _folder_has_required_modalities(folder_path):
    """Return True when folder contains FLAIR, T1, T1ce, and T2 volumes."""
    if not os.path.isdir(folder_path):
        return False

    found = set()
    try:
        for item in os.listdir(folder_path):
            modality = _infer_modality_from_filename(item)
            if modality is not None:
                found.add(modality)
    except Exception:
        return False

    return {'flair', 't1', 't1ce', 't2'}.issubset(found)


def _find_patient_dirs_with_modalities_local(root_path):
    """Fallback patient-dir finder used when Cell 11 helper is unavailable."""
    if not isinstance(root_path, str) or not os.path.isdir(root_path):
        return []

    patient_dirs = []
    if _folder_has_required_modalities(root_path):
        patient_dirs.append(root_path)

    # Primary layout: root/<patient_id>/<modality files>
    try:
        for item in sorted(os.listdir(root_path)):
            item_path = os.path.join(root_path, item)
            if not os.path.isdir(item_path):
                continue

            if _folder_has_required_modalities(item_path):
                patient_dirs.append(item_path)
                continue

            # Common nested layout: root/<split>/<patient_id>/<files>
            try:
                for sub in sorted(os.listdir(item_path)):
                    sub_path = os.path.join(item_path, sub)
                    if os.path.isdir(sub_path) and _folder_has_required_modalities(sub_path):
                        patient_dirs.append(sub_path)
            except Exception:
                continue
    except Exception:
        pass

    return [p for i, p in enumerate(patient_dirs) if p and p not in patient_dirs[:i]]


def _candidate_module_roots():
    """Collect likely roots where Cell 11 module files may exist."""
    roots = []

    if '__file__' in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))

    roots.extend([os.getcwd(), '/kaggle/working', '/kaggle/input', '/kaggle/input/datasets'])

    # Add one and two levels under common Kaggle mount points.
    for base in ['/kaggle/working', '/kaggle/input', '/kaggle/input/datasets', os.getcwd()]:
        if not os.path.isdir(base):
            continue
        try:
            with os.scandir(base) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    roots.append(entry.path)
                    try:
                        with os.scandir(entry.path) as sub_it:
                            for sub in sub_it:
                                if sub.is_dir():
                                    roots.append(sub.path)
                    except PermissionError:
                        continue
        except PermissionError:
            continue

    # Preserve order while deduplicating.
    return [p for i, p in enumerate(roots) if p and p not in roots[:i]]


def _discover_cell11_paths():
    """Find Cell 11 script paths across notebook/script/Kaggle layouts."""
    candidate_files = ['cell_11_inference_FIXED.py', 'cell_11_inference.py']
    paths = []

    # Fast direct checks under likely roots.
    for root in _candidate_module_roots():
        for name in candidate_files:
            p = os.path.join(root, name)
            if os.path.exists(p):
                paths.append(p)

    # Bounded recursive fallback for deeper Kaggle dataset nesting.
    max_depth = 5
    for base in ['/kaggle/working', '/kaggle/input', os.getcwd()]:
        if not os.path.isdir(base):
            continue

        base_norm = base.replace('\\', '/').rstrip('/')
        base_depth = base_norm.count('/')
        for root, dirs, files in os.walk(base):
            root_norm = root.replace('\\', '/').rstrip('/')
            depth = root_norm.count('/') - base_depth
            if depth > max_depth:
                dirs[:] = []
                continue

            for name in candidate_files:
                if name in files:
                    paths.append(os.path.join(root, name))

    # Preserve order while deduplicating.
    return [p for i, p in enumerate(paths) if p and p not in paths[:i]]


def _discover_npz_patch_files(path):
    """Collect NPZ patch files from a file path or directory root."""
    if not isinstance(path, str) or not path:
        return []

    if os.path.isfile(path):
        return [path] if path.lower().endswith('.npz') else []

    if not os.path.isdir(path):
        return []

    files = []
    max_depth = 6
    base_norm = path.replace('\\', '/').rstrip('/')
    base_depth = base_norm.count('/')
    for root, dirs, names in os.walk(path):
        root_norm = root.replace('\\', '/').rstrip('/')
        depth = root_norm.count('/') - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue

        for name in names:
            if name.lower().endswith('.npz'):
                files.append(os.path.join(root, name))

    patch_files = [p for p in files if 'patch' in os.path.basename(p).lower()]
    files = patch_files or files

    max_files = int(os.environ.get('CELL13_NPZ_MAX_FILES', '0'))
    if max_files > 0:
        files = files[:max_files]

    return [p for i, p in enumerate(sorted(files)) if p and p not in sorted(files)[:i]]


def _coerce_npz_image_layout(image, npz_path):
    """Normalize NPZ image array to CHWD with C=4 channels."""
    if image.ndim != 4:
        raise ValueError(f"NPZ image must be rank-4 in {npz_path}; got {image.shape}")

    if image.shape[0] == 4:
        image_chwd = image
    elif image.shape[-1] == 4:
        image_chwd = np.transpose(image, (3, 0, 1, 2))
    else:
        raise ValueError(
            f"Could not infer channel axis in {npz_path}; image shape={image.shape}. "
            "Expected channel-first or channel-last with 4 channels."
        )

    return image_chwd.astype(np.float32)


def _reorder_npz_channels_to_model(image_chwd):
    """Map NPZ channel order to model order: [flair, t1, t1ce, t2]."""
    channel_order_raw = os.environ.get('BRATS_NPZ_CHANNEL_ORDER', 't1,t1ce,t2,flair')
    order = [token.strip().lower() for token in channel_order_raw.split(',') if token.strip()]
    expected = {'flair', 't1', 't1ce', 't2'}
    if set(order) != expected:
        order = ['t1', 't1ce', 't2', 'flair']

    channel_index = {name: idx for idx, name in enumerate(order)}
    return image_chwd[
        [
            channel_index['flair'],
            channel_index['t1'],
            channel_index['t1ce'],
            channel_index['t2'],
        ]
    ]


def _normalize_chwd_01(image_chwd):
    """Min-max normalize each channel independently to [0,1]."""
    out = np.zeros_like(image_chwd, dtype=np.float32)
    for c in range(image_chwd.shape[0]):
        channel = image_chwd[c].astype(np.float32)
        c_min = float(np.min(channel))
        c_max = float(np.max(channel))
        if np.isfinite(c_min) and np.isfinite(c_max) and c_max > c_min:
            out[c] = (channel - c_min) / (c_max - c_min)
        else:
            out[c] = 0.0
    return out


def _resolve_active_threshold(load_model_fn):
    """Resolve inference threshold from Cell 11 globals or environment."""
    with np.errstate(all='ignore'):
        try:
            loader_globals = getattr(load_model_fn, '__globals__', {})
            if isinstance(loader_globals, dict) and 'CONFIDENCE_THRESHOLD' in loader_globals:
                return float(loader_globals['CONFIDENCE_THRESHOLD'])
        except Exception:
            pass

    for key in ('BRATS_INFERENCE_THRESHOLD', 'CONFIDENCE_THRESHOLD', 'BRATS_THRESHOLD'):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except Exception:
            continue

    return 0.5


def _predict_single_npz_patch(npz_path, model, output_dir, threshold):
    """Run inference for a single NPZ patch file and save predicted mask NPZ."""
    with np.load(npz_path) as data:
        if 'image' not in data:
            raise KeyError(f"NPZ missing required key 'image': {npz_path}")
        image = data['image']

    image_chwd = _coerce_npz_image_layout(image, npz_path)
    image_chwd = _reorder_npz_channels_to_model(image_chwd)
    image_chwd = _normalize_chwd_01(image_chwd)

    # Model expects NHWC slices, where N is depth axis.
    x = np.transpose(image_chwd, (3, 1, 2, 0)).astype(np.float32)
    if x.ndim != 4 or x.shape[-1] != 4:
        raise RuntimeError(f"Unexpected NPZ inference tensor shape: {x.shape}")

    infer_batch = int(os.environ.get('INFERENCE_BATCH_SIZE', '8'))
    y_prob = model.predict(x, batch_size=max(1, infer_batch), verbose=0).astype(np.float32)
    if y_prob.ndim != 4:
        raise RuntimeError(f"Unexpected prediction shape for {npz_path}: {y_prob.shape}")
    if y_prob.shape[-1] != 1:
        y_prob = np.max(y_prob, axis=-1, keepdims=True)

    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob_hwd = np.transpose(np.clip(y_prob[..., 0], 0.0, 1.0), (1, 2, 0)).astype(np.float32)
    y_bin = (y_prob >= float(threshold)).astype(np.uint8)
    mask_hwd = np.transpose(y_bin[..., 0], (1, 2, 0))

    patch_id = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_npz_path = os.path.join(output_dir, f'{patch_id}_predicted_mask.npz')
    np.savez_compressed(
        output_npz_path,
        mask=mask_hwd,
        threshold=float(threshold),
        source=npz_path,
    )

    counts = {
        'tumor': int(np.sum(mask_hwd > 0)),
        'output_npz': output_npz_path,
    }

    if os.environ.get('CELL13_SAVE_PROB_NPZ', '0') == '1':
        output_prob_npz_path = os.path.join(output_dir, f'{patch_id}_predicted_prob.npz')
        np.savez_compressed(
            output_prob_npz_path,
            prob=prob_hwd,
            threshold=float(threshold),
            source=npz_path,
        )
        counts['output_prob_npz'] = output_prob_npz_path

    if os.environ.get('CELL13_GENERATE_VISUALS', '1') == '1':
        try:
            visual_info = _create_npz_overlay_viewer(
                npz_path,
                mask_hwd,
                prob_hwd,
                output_dir,
                threshold=float(threshold),
            )
            if isinstance(visual_info, dict):
                counts.update(visual_info)
        except Exception as exc:
            counts['visual_error'] = str(exc)
            print(f"[WARN] Could not build NPZ overlay viewer for {npz_path}: {exc}")

    return mask_hwd, counts


def _normalize_uint8_slice(slice_2d):
    """Normalize 2D slice to uint8 [0, 255] for visualization."""
    arr = np.asarray(slice_2d, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def _overlay_mask_on_gray(gray_uint8, mask_2d, alpha=0.45):
    """Create RGB overlay image with red mask on grayscale base."""
    base = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1).astype(np.float32) / 255.0
    mask_bool = np.asarray(mask_2d, dtype=np.uint8) > 0
    alpha = float(np.clip(alpha, 0.0, 1.0))

    base_r = base[..., 0]
    base_g = base[..., 1]
    base_b = base[..., 2]

    base_r[mask_bool] = (1.0 - alpha) * base_r[mask_bool] + alpha * 1.0
    base_g[mask_bool] = (1.0 - alpha) * base_g[mask_bool] + alpha * 0.0
    base_b[mask_bool] = (1.0 - alpha) * base_b[mask_bool] + alpha * 0.0

    rgb = np.stack([base_r, base_g, base_b], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _coerce_volume_to_hwd(volume, reference_shape, volume_name):
    """Coerce a 3D or singleton-4D volume into HWD matching reference shape."""
    arr = np.asarray(volume)

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # If multi-channel, keep first channel as fallback.
            arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[..., 0]

    if arr.ndim != 3:
        return None

    reference_shape = tuple(int(v) for v in reference_shape)
    if tuple(arr.shape) == reference_shape:
        return arr

    for axes in itertools.permutations((0, 1, 2), 3):
        candidate = np.transpose(arr, axes)
        if tuple(candidate.shape) == reference_shape:
            return candidate

    print(f"[WARN] Could not align {volume_name} to expected shape {reference_shape}; got {arr.shape}")
    return None


def _extract_ground_truth_hwd(npz_data, reference_shape):
    """Return binary ground-truth HWD volume and key name when present in NPZ."""
    gt_keys = ('mask', 'seg', 'label', 'labels', 'target', 'y', 'gt', 'ground_truth')

    for key in gt_keys:
        if key not in npz_data:
            continue

        gt_volume = _coerce_volume_to_hwd(npz_data[key], reference_shape, f'ground truth key={key}')
        if gt_volume is None:
            continue

        gt_binary = (np.nan_to_num(gt_volume, nan=0.0, posinf=0.0, neginf=0.0) > 0).astype(np.uint8)
        return gt_binary, key

    return None, None


def _pick_visual_slice_indices(mask_hwd, top_k):
    """Select slice indices for visualization, prioritizing tumor-heavy slices."""
    if mask_hwd.ndim != 3:
        raise ValueError(f"Expected mask shape HWD, got {mask_hwd.shape}")

    depth = int(mask_hwd.shape[2])
    if depth <= 0:
        return [], []

    top_k = max(1, min(int(top_k), depth))
    tumor_scores = [int(np.sum(mask_hwd[:, :, z] > 0)) for z in range(depth)]
    nonzero = [z for z, score in enumerate(tumor_scores) if score > 0]

    if nonzero:
        ranked = sorted(nonzero, key=lambda z: tumor_scores[z], reverse=True)[:top_k]
        return sorted(ranked), tumor_scores

    # Fallback: evenly spaced slices when there is no predicted tumor.
    lin = np.linspace(0, depth - 1, num=top_k)
    selected = []
    for value in lin:
        z = int(round(float(value)))
        if z not in selected:
            selected.append(z)
    return selected, tumor_scores


def _write_npz_viewer_html(html_path, slices, source_npz, modality, alpha, threshold, gt_key):
    """Write a standalone 4-panel HTML viewer for NPZ inference inspection."""
    payload = json.dumps(slices)
    source_name = os.path.basename(source_npz)
    gt_text = gt_key if isinstance(gt_key, str) and gt_key else 'not available'
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Cell 13 NPZ Overlay Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #222; background: #f7f8fa; }}
    h1 {{ font-size: 18px; margin: 0 0 8px; }}
    .meta {{ margin: 0 0 12px; font-size: 13px; color: #444; }}
    .controls {{ display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    .label {{ font-weight: 600; }}
    .row {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{ background: #fff; border: 1px solid #d8dde6; border-radius: 8px; padding: 10px; }}
    .card h2 {{ margin: 0 0 8px; font-size: 14px; }}
    img {{ width: 100%; height: auto; image-rendering: pixelated; border-radius: 6px; border: 1px solid #e3e7ee; }}
  </style>
</head>
<body>
    <h1>Cell 13 Segmentation 4-Panel Viewer</h1>
    <p class=\"meta\">Source: {source_name} | Modality: {modality} | Threshold: {threshold:.2f} | Overlay alpha: {alpha:.2f} | GT key: {gt_text}</p>
  <div class=\"controls\">
    <span class=\"label\">Slice:</span>
    <input id=\"slice\" type=\"range\" min=\"0\" max=\"{max(0, len(slices) - 1)}\" value=\"0\" step=\"1\" />
    <span id=\"sliceMeta\"></span>
  </div>
  <div class=\"row\">
        <div class=\"card\"><h2>MRI ({modality.upper()})</h2><img id=\"flair\" alt=\"flair\" /></div>
        <div class=\"card\"><h2>Ground Truth</h2><img id=\"gt\" alt=\"ground_truth\" /></div>
        <div class=\"card\"><h2>Prediction Probability</h2><img id=\"prob\" alt=\"probability\" /></div>
        <div class=\"card\"><h2>Overlay</h2><img id=\"overlay\" alt=\"overlay\" /></div>
  </div>
  <script>
    const slices = {payload};
    const slider = document.getElementById('slice');
        const flair = document.getElementById('flair');
        const gt = document.getElementById('gt');
        const prob = document.getElementById('prob');
    const overlay = document.getElementById('overlay');
    const meta = document.getElementById('sliceMeta');

    function render(idx) {{
      if (!slices.length) {{
        meta.textContent = 'No slices available';
        return;
      }}
      const s = slices[idx];
            flair.src = s.flair;
            gt.src = s.ground_truth;
            prob.src = s.probability;
      overlay.src = s.overlay;
      meta.textContent = `z=${{s.z}} | tumor_voxels=${{s.tumor_voxels}}`;
    }}

    slider.addEventListener('input', () => render(Number(slider.value)));
    render(0);
  </script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _create_npz_overlay_viewer(npz_path, mask_hwd, prob_hwd, output_dir, threshold):
    """Build 4-panel PNG slices and HTML viewer for NPZ prediction output."""
    if Image is None:
        raise ImportError('Pillow is not available. Install pillow or set CELL13_GENERATE_VISUALS=0.')

    if mask_hwd.ndim != 3:
        raise ValueError(f"Expected mask shape HWD, got {mask_hwd.shape}")
    if prob_hwd.ndim != 3:
        raise ValueError(f"Expected probability shape HWD, got {prob_hwd.shape}")
    if prob_hwd.shape != mask_hwd.shape:
        raise ValueError(f"Mask/probability shape mismatch: {mask_hwd.shape} vs {prob_hwd.shape}")

    with np.load(npz_path) as data:
        if 'image' not in data:
            raise KeyError(f"NPZ missing required key 'image': {npz_path}")
        image = data['image']
        gt_hwd, gt_key = _extract_ground_truth_hwd(data, mask_hwd.shape)

    image_chwd = _coerce_npz_image_layout(image, npz_path)
    image_chwd = _reorder_npz_channels_to_model(image_chwd)
    image_chwd = _normalize_chwd_01(image_chwd)

    modality = os.environ.get('CELL13_VIS_MODALITY', 'flair').strip().lower()
    channel_map = {'flair': 0, 't1': 1, 't1ce': 2, 't2': 3}
    channel_index = channel_map.get(modality, 0)
    modality = [k for k, v in channel_map.items() if v == channel_index][0]

    alpha = float(os.environ.get('CELL13_VIS_ALPHA', '0.45'))
    top_k = int(os.environ.get('CELL13_VIS_TOPK', '8'))
    selected_slices, tumor_scores = _pick_visual_slice_indices(mask_hwd, top_k)

    visual_dir = os.path.join(output_dir, 'visuals')
    os.makedirs(visual_dir, exist_ok=True)

    records = []
    for z in selected_slices:
        base_slice = image_chwd[channel_index, :, :, z]
        flair_uint8 = _normalize_uint8_slice(base_slice)
        prob_uint8 = np.clip(np.round(np.clip(prob_hwd[:, :, z], 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
        pred_mask_uint8 = (mask_hwd[:, :, z] > 0).astype(np.uint8) * 255
        gt_uint8 = (gt_hwd[:, :, z] * 255).astype(np.uint8) if gt_hwd is not None else np.zeros_like(flair_uint8, dtype=np.uint8)
        overlay_uint8 = _overlay_mask_on_gray(flair_uint8, pred_mask_uint8 > 0, alpha=alpha)

        overlay_name = f'overlay_z{z:03d}.png'
        flair_name = f'{modality}_z{z:03d}.png'
        gt_name = f'gt_z{z:03d}.png'
        prob_name = f'prob_z{z:03d}.png'

        overlay_path = os.path.join(visual_dir, overlay_name)
        flair_path = os.path.join(visual_dir, flair_name)
        gt_path = os.path.join(visual_dir, gt_name)
        prob_path = os.path.join(visual_dir, prob_name)

        Image.fromarray(overlay_uint8).save(overlay_path)
        Image.fromarray(flair_uint8).save(flair_path)
        Image.fromarray(gt_uint8).save(gt_path)
        Image.fromarray(prob_uint8).save(prob_path)

        records.append(
            {
                'z': int(z),
                'tumor_voxels': int(tumor_scores[z]) if z < len(tumor_scores) else 0,
                'overlay': overlay_name,
                'flair': flair_name,
                'ground_truth': gt_name,
                'probability': prob_name,
            }
        )

    viewer_html = os.path.join(visual_dir, 'viewer.html')
    _write_npz_viewer_html(
        viewer_html,
        records,
        npz_path,
        modality,
        alpha,
        threshold=float(threshold),
        gt_key=gt_key,
    )

    viewer_data_path = os.path.join(visual_dir, 'viewer_data.json')
    with open(viewer_data_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'source_npz': npz_path,
                'modality': modality,
                'threshold': float(threshold),
                'overlay_alpha': float(alpha),
                'ground_truth_key': gt_key if gt_key is not None else 'not_available',
                'slices': records,
            },
            f,
            indent=2,
        )

    return {
        'viewer_html': viewer_html,
        'visual_dir': visual_dir,
        'visual_slice_count': len(records),
        'ground_truth_key': gt_key if gt_key is not None else 'not_available',
        'viewer_data_json': viewer_data_path,
    }


def _display_inline_visual_panels(summary):
    """Render a Cell9-style 4-panel preview inline when running in notebooks."""
    if os.environ.get('CELL13_INLINE_VISUALS', '1') != '1':
        return

    results = summary.get('results')
    if not isinstance(results, dict) or not results:
        return

    if Image is None:
        return

    try:
        from IPython.display import display, Markdown
        import matplotlib.pyplot as plt
    except Exception:
        return

    max_items_raw = os.environ.get('CELL13_INLINE_MAX_ITEMS', '2')
    try:
        max_items = max(1, int(max_items_raw))
    except Exception:
        max_items = 2

    shown = 0
    for item_id in sorted(results.keys()):
        if shown >= max_items:
            break

        info = results[item_id]
        if not isinstance(info, dict) or 'error' in info:
            continue

        visual_dir = info.get('visual_dir')
        viewer_data_json = info.get('viewer_data_json')
        if not isinstance(visual_dir, str) or not os.path.isdir(visual_dir):
            continue

        if not isinstance(viewer_data_json, str) or not os.path.isfile(viewer_data_json):
            viewer_data_json = os.path.join(visual_dir, 'viewer_data.json')
            if not os.path.isfile(viewer_data_json):
                continue

        try:
            with open(viewer_data_json, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception:
            continue

        slices = payload.get('slices')
        if not isinstance(slices, list) or len(slices) == 0:
            continue

        best_slice = max(slices, key=lambda s: int(s.get('tumor_voxels', 0)))

        def _resolve_png(name_key):
            name = best_slice.get(name_key)
            if not isinstance(name, str) or not name:
                return None
            p = os.path.join(visual_dir, name)
            return p if os.path.isfile(p) else None

        flair_path = _resolve_png('flair')
        gt_path = _resolve_png('ground_truth')
        prob_path = _resolve_png('probability')
        overlay_path = _resolve_png('overlay')

        if not all([flair_path, gt_path, prob_path, overlay_path]):
            continue

        try:
            flair_img = np.array(Image.open(flair_path))
            gt_img = np.array(Image.open(gt_path))
            prob_img = np.array(Image.open(prob_path))
            overlay_img = np.array(Image.open(overlay_path))
        except Exception:
            continue

        z = int(best_slice.get('z', 0))
        tumor_voxels = int(best_slice.get('tumor_voxels', 0))
        modality = str(payload.get('modality', 'flair')).upper()

        display(Markdown(f"### Cell 13 Inline Preview: {item_id} (z={z}, tumor_voxels={tumor_voxels:,})"))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=120)
        axes[0].imshow(flair_img, cmap='gray')
        axes[0].set_title(f'MRI ({modality})')

        axes[1].imshow(gt_img, cmap='gray')
        axes[1].set_title('Ground Truth')

        axes[2].imshow(prob_img, cmap='inferno', vmin=0, vmax=255)
        axes[2].set_title('Prediction Probability')

        axes[3].imshow(overlay_img)
        axes[3].set_title('Overlay')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        shown += 1


def _predict_multiple_npz_patches(npz_files, model, output_base_dir, threshold):
    """Run inference for multiple NPZ patches and aggregate counts/errors."""
    results = {}
    for npz_path in npz_files:
        patch_id = os.path.splitext(os.path.basename(npz_path))[0]
        out_dir = os.path.join(output_base_dir, patch_id)
        try:
            _, counts = _predict_single_npz_patch(npz_path, model, out_dir, threshold)
            results[patch_id] = counts
        except Exception as exc:
            results[patch_id] = {'error': str(exc)}
    return results


def _resolve_inference_functions():
    """Resolve Cell 11 inference functions across script and notebook workflows."""
    required_core = [
        'load_inference_model',
        'predict_multiple_patients',
        'predict_patient',
    ]
    optional_finder = 'find_patient_dirs_with_modalities'
    load_errors = []
    candidate_paths = []

    # Case 1: Cell 11 was executed in the same notebook/session.
    if all(name in globals() and callable(globals()[name]) for name in required_core):
        finder_fn = globals().get(optional_finder, _find_patient_dirs_with_modalities_local)
        return (
            finder_fn,
            globals()['load_inference_model'],
            globals()['predict_multiple_patients'],
            globals()['predict_patient'],
        )

    # Case 2: Normal Python import works.
    for module_name in ('cell_11_inference_FIXED', 'cell_11_inference'):
        try:
            module = __import__(
                module_name,
                fromlist=[
                    'load_inference_model',
                    'predict_multiple_patients',
                    'predict_patient',
                ],
            )
            if all(hasattr(module, name) for name in required_core):
                finder_fn = getattr(module, optional_finder, _find_patient_dirs_with_modalities_local)
                return (
                    finder_fn,
                    getattr(module, 'load_inference_model'),
                    getattr(module, 'predict_multiple_patients'),
                    getattr(module, 'predict_patient'),
                )
        except ModuleNotFoundError as exc:
            # If another dependency is missing inside Cell 11, keep the detail.
            if getattr(exc, 'name', '') not in {module_name, ''}:
                load_errors.append(f'{module_name}: {exc}')
            continue
        except Exception as exc:
            load_errors.append(f'{module_name}: {exc}')
            continue

    # Case 3: Load module directly from file if present (Kaggle/Notebook fallback).
    candidate_paths = _discover_cell11_paths()

    for module_path in candidate_paths:
        try:
            module_tag = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(module_tag, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if all(hasattr(module, name) for name in required_core):
                    finder_fn = getattr(module, optional_finder, _find_patient_dirs_with_modalities_local)
                    return (
                        finder_fn,
                        getattr(module, 'load_inference_model'),
                        getattr(module, 'predict_multiple_patients'),
                        getattr(module, 'predict_patient'),
                    )
        except Exception as exc:
            load_errors.append(f'{module_path}: {exc}')

    details = []
    if candidate_paths:
        preview = candidate_paths[:5]
        more = ' ...' if len(candidate_paths) > 5 else ''
        details.append(f'checked paths={preview}{more}')
    if load_errors:
        details.append(f'last errors={load_errors[-3:]}')
    suffix = f' [{"; ".join(details)}]' if details else ''

    raise ModuleNotFoundError(
        "Could not load Cell 11 inference helpers. Run Cell 11 first in the same notebook, "
        "or ensure cell_11_inference_FIXED.py (or legacy cell_11_inference.py) exists in the working directory."
        f"{suffix}"
    )

# ========================
# DEFAULT VARIABLES
# ========================
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd())
if 'RESULTS_DIR' not in globals():
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================
# USER CONFIG
# ========================
# Set this to your Kaggle upload/input root directory.
UPLOAD_ROOT = os.environ.get('UPLOAD_ROOT', '/kaggle/input/datasets/hesh2094/cell-13')

# Mode options: 'auto', 'single', 'batch'
RUN_MODE = os.environ.get('CELL13_RUN_MODE', 'single').strip().lower()

# Required only when RUN_MODE = 'single'
SINGLE_PATIENT_DIR = os.environ.get(
    'SINGLE_PATIENT_DIR',
    '/kaggle/input/datasets/hesh2094/cell-13/BraTS-GLI-00020-000_patch_0.npz',
)

# Optional output summary JSON
SAVE_SUMMARY_JSON = True
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, 'cell13_upload_inference_summary.json')
PREDICTION_OUTPUT_BASE = os.path.join(RESULTS_DIR, 'predictions')
os.makedirs(PREDICTION_OUTPUT_BASE, exist_ok=True)


def _auto_detect_upload_root(finder_fn):
    """Find a Kaggle input root that contains at least one valid patient folder."""
    candidates = []

    def _looks_like_training_path(path):
        """Heuristic: avoid selecting training corpora as upload input roots."""
        p = str(path).replace('\\', '/').lower()
        training_tokens = [
            'training',
            'miccai_brats2020_trainingdata',
            'brats2021_training_data',
            'brats_2019_data_training',
            'brats20_training',
        ]
        return any(token in p for token in training_tokens)

    # Highest priority: reuse already-resolved path from earlier cells.
    # Skip if it is likely the training corpus; Cell 13 should target user uploads.
    train_path = globals().get('TRAIN_PATH')
    if (
        isinstance(train_path, str)
        and os.path.isdir(train_path)
        and not _looks_like_training_path(train_path)
    ):
        candidates.append(train_path)

    # Common Kaggle locations: direct datasets and nested owner datasets.
    for base in ['/kaggle/input', '/kaggle/input/datasets', '/kaggle/working', os.getcwd()]:
        if not os.path.isdir(base):
            continue

        for item in sorted(os.listdir(base)):
            item_path = os.path.join(base, item)
            if not os.path.isdir(item_path):
                continue

            # Check dataset root itself.
            candidates.append(item_path)

            # Check one level deeper (many datasets store data in a subfolder).
            for sub in sorted(os.listdir(item_path)):
                sub_path = os.path.join(item_path, sub)
                if os.path.isdir(sub_path):
                    candidates.append(sub_path)

    # Recursive walk fallback for deeply nested Kaggle dataset structures.
    max_depth = 6
    for base in ['/kaggle/input', '/kaggle/input/datasets', os.getcwd()]:
        if not os.path.isdir(base):
            continue

        base_depth = base.rstrip('/').count('/')
        for root, dirs, _ in os.walk(base):
            depth = root.rstrip('/').count('/') - base_depth
            if depth > max_depth:
                dirs[:] = []
                continue
            candidates.append(root)

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)

        if _looks_like_training_path(root):
            continue

        try:
            patient_dirs = finder_fn(root)
        except Exception:
            continue

        if len(patient_dirs) > 0:
            return root, len(patient_dirs)

        # NPZ patch fallback for datasets that are not patient-folder NIfTI layouts.
        try:
            npz_files = _discover_npz_patch_files(root)
        except Exception:
            npz_files = []
        if len(npz_files) > 0:
            return root, len(npz_files)

    return None, 0


def _to_json_safe(value):
    """Recursively convert values to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    # NumPy scalars (e.g., int64/float32) typically expose .item().
    item_fn = getattr(value, 'item', None)
    if callable(item_fn):
        try:
            return _to_json_safe(item_fn())
        except Exception:
            pass

    # NumPy arrays and similar objects typically expose .tolist().
    tolist_fn = getattr(value, 'tolist', None)
    if callable(tolist_fn):
        try:
            return _to_json_safe(tolist_fn())
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Fallback: keep pipeline robust even with unexpected custom objects.
    return str(value)


def _extract_output_path(info):
    """Return best-known output path from a per-item result payload."""
    if not isinstance(info, dict):
        return None

    for key in ('output_npz', 'output_nifti', 'output_path', 'viewer_html', 'visual_dir'):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value

    return None


def _aggregate_results(results):
    """Compute compact aggregate metrics for final console and JSON summary."""
    aggregate = {
        'total_items': 0,
        'success_items': 0,
        'error_items': 0,
        'all_zero_items': 0,
        'total_tumor_voxels': 0,
    }

    if not isinstance(results, dict):
        return aggregate

    aggregate['total_items'] = len(results)
    for info in results.values():
        if isinstance(info, dict) and 'error' in info:
            aggregate['error_items'] += 1
            continue

        aggregate['success_items'] += 1
        if isinstance(info, dict):
            tumor = int(info.get('tumor', 0))
            aggregate['total_tumor_voxels'] += tumor
            if bool(info.get('all_zero')):
                aggregate['all_zero_items'] += 1

    return aggregate


def _print_final_summary(summary, summary_json_path=None):
    """Print a cleaner, dashboard-style final summary."""
    results = summary.get('results')
    aggregate = _aggregate_results(results)

    print('\n' + '=' * 70)
    print('FINAL OUTPUT SUMMARY')
    print('=' * 70)
    print(f"Status: {summary.get('status', 'unknown')}")
    print(f"Run mode: {summary.get('run_mode')}")
    print(f"Upload root: {summary.get('upload_root')}")
    print(f"Threshold: {float(summary.get('threshold', 0.5)):.2f}")
    print(f"Predictions base: {summary.get('predictions_base')}")
    print(
        f"Processed={aggregate['total_items']} | "
        f"Success={aggregate['success_items']} | "
        f"Errors={aggregate['error_items']} | "
        f"All-zero={aggregate['all_zero_items']}"
    )
    print(f"Total tumor voxels: {aggregate['total_tumor_voxels']:,}")

    elapsed_seconds = summary.get('elapsed_seconds')
    if elapsed_seconds is not None:
        print(f"Elapsed: {float(elapsed_seconds):.2f}s")

    if summary_json_path:
        print(f"Summary JSON: {summary_json_path}")

    if isinstance(results, dict):
        print('-' * 70)
        for item_id in sorted(results.keys()):
            info = results[item_id]
            if isinstance(info, dict) and 'error' in info:
                print(f"  [ERROR] {item_id}: {info['error']}")
                continue

            tumor = int(info.get('tumor', 0)) if isinstance(info, dict) else 0
            all_zero = bool(info.get('all_zero')) if isinstance(info, dict) else False
            zero_suffix = ' [ALL_ZERO]' if all_zero else ''
            output_path = _extract_output_path(info)
            output_suffix = f" | out={output_path}" if output_path else ''
            viewer_html = info.get('viewer_html') if isinstance(info, dict) else None
            viewer_suffix = f" | viewer={viewer_html}" if isinstance(viewer_html, str) and viewer_html else ''
            gt_key = info.get('ground_truth_key') if isinstance(info, dict) else None
            gt_suffix = f" | gt={gt_key}" if isinstance(gt_key, str) and gt_key else ''
            visual_error = info.get('visual_error') if isinstance(info, dict) else None
            visual_warn = f" | visual_warn={visual_error}" if isinstance(visual_error, str) and visual_error else ''
            print(f"  [OK] {item_id}: tumor={tumor:,}{zero_suffix}{output_suffix}{viewer_suffix}{gt_suffix}{visual_warn}")
    else:
        print(f"Raw results: {results}")

    print('=' * 70)


def run_cell_13_upload_predict():
    """Run upload-to-segmentation at the final stage of the pipeline."""
    start_time = time.time()

    (
        find_patient_dirs_with_modalities,
        load_inference_model,
        predict_multiple_patients,
        predict_patient,
    ) = _resolve_inference_functions()

    finder_origin = 'cell11' if find_patient_dirs_with_modalities.__name__ != '_find_patient_dirs_with_modalities_local' else 'cell13_fallback'

    print('=' * 70)
    print('CELL 13: FINAL UPLOAD -> PREDICT RUNNER')
    print(f'Patient folder finder: {finder_origin}')
    print('=' * 70)

    if UPLOAD_ROOT.rstrip('/').rstrip('\\') == '/kaggle/input/your-upload-dataset-root':
        detected_root, detected_count = _auto_detect_upload_root(find_patient_dirs_with_modalities)
        if detected_root:
            print(
                "⚠ UPLOAD_ROOT is placeholder. "
                f"Auto-detected: {detected_root} ({detected_count} patient folders)"
            )
            active_upload_root = detected_root

            # Safety guard: prevent accidental full-dataset inference in auto mode.
            if RUN_MODE == 'auto' and detected_count > 20:
                print('✗ Auto-detected root appears to be a large dataset.')
                print("  To avoid accidental multi-hour batch inference, set one of:")
                print("  1) UPLOAD_ROOT to your uploaded patient folder/dataset")
                print("  2) RUN_MODE='single' and SINGLE_PATIENT_DIR='/path/to/patient'")
                print("  3) RUN_MODE='batch' explicitly (if you really want full batch)")
                return None
        else:
            print('✗ UPLOAD_ROOT is still placeholder and no valid input root was auto-detected.')
            print("  Set UPLOAD_ROOT to your Kaggle dataset folder and re-run Cell 13.")
            print("  Example: /kaggle/input/your-dataset-name")
            return None
    else:
        active_upload_root = UPLOAD_ROOT

    model = load_inference_model()
    if model is None:
        print('✗ Stopped: no trained model available.')
        return None

    active_threshold = _resolve_active_threshold(load_inference_model)
    print(f'Using inference threshold: {active_threshold:.2f}')

    if RUN_MODE not in {'auto', 'single', 'batch'}:
        print(f"✗ Invalid RUN_MODE: {RUN_MODE}")
        print("  Allowed values: 'auto', 'single', 'batch'")
        return None

    summary = {
        'run_mode': RUN_MODE,
        'upload_root': active_upload_root,
        'threshold': float(active_threshold),
        'predictions_base': PREDICTION_OUTPUT_BASE,
        'status': 'failed',
        'results': None,
    }

    def _attach_zero_flag(patient_id, counts, output_path=None, output_key='output_path'):
        """Add all-zero prediction flag and print warning when mask is empty."""
        if not isinstance(counts, dict):
            return counts

        if 'error' in counts:
            return counts

        tumor_count = int(counts.get('tumor', 0))
        all_zero = (tumor_count == 0)

        updated = dict(counts)
        updated['tumor'] = tumor_count
        updated['all_zero'] = all_zero

        if isinstance(output_path, str) and output_path:
            updated[output_key] = output_path

        if all_zero:
            print(
                f"⚠ All-zero prediction for '{patient_id}'. "
                "Check modality completeness/orientation or threshold settings."
            )

        return updated

    if RUN_MODE == 'single':
        if not SINGLE_PATIENT_DIR:
            print("✗ SINGLE_PATIENT_DIR is empty. Set it when RUN_MODE='single'.")
            return None

        # Allow SINGLE_PATIENT_DIR to be either patient folder or NPZ patch path.
        single_npz_files = _discover_npz_patch_files(SINGLE_PATIENT_DIR)
        if single_npz_files:
            if len(single_npz_files) > 1:
                print(
                    f"⚠ RUN_MODE='single' got {len(single_npz_files)} NPZ files. "
                    "Using the first one."
                )
            npz_path = single_npz_files[0]
            patch_name = os.path.splitext(os.path.basename(npz_path))[0]
            output_dir = os.path.join(PREDICTION_OUTPUT_BASE, patch_name)
            volume, counts = _predict_single_npz_patch(npz_path, model, output_dir, active_threshold)
            if volume is None:
                print('✗ Single-NPZ inference failed.')
                return None

            summary['status'] = 'ok'
            summary['results'] = {
                patch_name: _attach_zero_flag(patch_name, counts),
            }
        else:
            patient_name = os.path.basename(SINGLE_PATIENT_DIR.rstrip(os.sep))
            output_dir = os.path.join(PREDICTION_OUTPUT_BASE, patient_name)
            volume, counts = predict_patient(SINGLE_PATIENT_DIR, model=model, output_dir=output_dir)
            if volume is None:
                print('✗ Single-patient inference failed.')
                return None

            summary['status'] = 'ok'
            summary['results'] = {
                patient_name: _attach_zero_flag(
                    patient_name,
                    counts,
                    output_path=os.path.join(output_dir, 'predicted_mask.nii.gz'),
                    output_key='output_nifti',
                ),
            }

    else:
        patient_dirs = find_patient_dirs_with_modalities(active_upload_root)

        if len(patient_dirs) == 0:
            npz_files = _discover_npz_patch_files(active_upload_root)
            if len(npz_files) == 0:
                print('✗ No valid patient folders found in upload root.')
                print('  Each patient folder must include FLAIR, T1, T1ce, T2 NIfTI files.')
                print('  Or provide NPZ patch files (*.npz) in upload root.')
                return None

            print(f'NPZ patch mode detected: {len(npz_files)} files')
            if RUN_MODE == 'auto' and len(npz_files) == 1:
                npz_path = npz_files[0]
                patch_name = os.path.splitext(os.path.basename(npz_path))[0]
                output_dir = os.path.join(PREDICTION_OUTPUT_BASE, patch_name)
                volume, counts = _predict_single_npz_patch(npz_path, model, output_dir, active_threshold)
                if volume is None:
                    print('✗ Auto single-NPZ inference failed.')
                    return None

                summary['status'] = 'ok'
                summary['results'] = {
                    patch_name: _attach_zero_flag(patch_name, counts),
                }
            else:
                npz_results = _predict_multiple_npz_patches(
                    npz_files,
                    model=model,
                    output_base_dir=PREDICTION_OUTPUT_BASE,
                    threshold=active_threshold,
                )
                enriched_results = {
                    patch_id: _attach_zero_flag(patch_id, info)
                    for patch_id, info in npz_results.items()
                }
                summary['status'] = 'ok'
                summary['results'] = enriched_results

        else:
            if RUN_MODE == 'auto' and len(patient_dirs) == 1:
                patient_dir = patient_dirs[0]
                patient_name = os.path.basename(patient_dir.rstrip(os.sep))
                output_dir = os.path.join(PREDICTION_OUTPUT_BASE, patient_name)
                volume, counts = predict_patient(patient_dir, model=model, output_dir=output_dir)
                if volume is None:
                    print('✗ Auto single-patient inference failed.')
                    return None

                summary['status'] = 'ok'
                summary['results'] = {
                    patient_name: _attach_zero_flag(
                        patient_name,
                        counts,
                        output_path=os.path.join(output_dir, 'predicted_mask.nii.gz'),
                        output_key='output_nifti',
                    ),
                }
            else:
                batch_results = predict_multiple_patients(
                    patient_dirs,
                    model=model,
                    # Cell 11 appends "predictions/<patient_id>" internally.
                    output_base_dir=RESULTS_DIR,
                )

                enriched_results = {}
                for patient_id, info in batch_results.items():
                    output_nifti_path = os.path.join(
                        RESULTS_DIR,
                        'predictions',
                        patient_id,
                        'predicted_mask.nii.gz',
                    )
                    enriched_results[patient_id] = _attach_zero_flag(
                        patient_id,
                        info,
                        output_path=output_nifti_path,
                        output_key='output_nifti',
                    )

                summary['status'] = 'ok'
                summary['results'] = enriched_results

    summary['elapsed_seconds'] = round(time.time() - start_time, 2)
    if summary['status'] == 'ok':
        summary['aggregate'] = _aggregate_results(summary['results'])

    saved_summary_path = None
    if SAVE_SUMMARY_JSON and summary['status'] == 'ok':
        summary_safe = _to_json_safe(summary)
        with open(SUMMARY_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary_safe, f, indent=2)
        saved_summary_path = SUMMARY_JSON_PATH
        print(f"✓ Saved summary: {SUMMARY_JSON_PATH}")

    _print_final_summary(summary, summary_json_path=saved_summary_path)

    if summary.get('status') == 'ok':
        _display_inline_visual_panels(summary)

    print('✓ Cell 13 complete. Upload inference finished.')
    print('=' * 70)

    return summary


if __name__ == '__main__':
    CELL13_RESULTS = run_cell_13_upload_predict()
