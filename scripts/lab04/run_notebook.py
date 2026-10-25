"""Execute Lab04, audit its results and export figures/metrics for the report."""

import base64
from pathlib import Path

import nbformat
from nbclient import NotebookClient

FIGURES = [
    "dataset_pairs",
    "rgb_channels",
    "baseline_hash",
    "wavelet_bands",
    "channel_distances",
    "hamming_matrix",
    "robustness",
    "threshold_selection",
    "roc_evaluation",
    "review_pairs",
    "wavelet_comparison",
    "top_k",
]

AUDIT = """
import json, platform, scipy, sklearn
assert np.array_equal(demo_hash, hashes[demo_id])
assert distance_matrix.shape == (len(sources), len(sources))
assert np.array_equal(distance_matrix, distance_matrix.T)
assert not np.diag(distance_matrix).any()
assert all(h.shape == (3 * params['bits'] ** 2,) and np.isin(h, [0, 1]).all() for h in hashes.values())
cal_sources = {s['source_id'] for s in sources if s['split'] == 'calibration'}
eval_sources = {s['source_id'] for s in sources if s['split'] == 'evaluation'}
assert cal_sources.isdisjoint(eval_sources)
for split, rows in pairs.items():
    for pair in rows:
        left, right = metadata[pair['left_id']], metadata[pair['right_id']]
        assert left['split'] == right['split'] == split
        assert (left['source_id'] == right['source_id']) == (pair['label'] == '1')
assert [d for _, d in top_results] == sorted(d for _, d in top_results)
assert all((root / s['path']).resolve() != query_path.resolve() for s in candidates)

themed = {}
for theme in dict.fromkeys(p['theme'] for p in pairs['evaluation']):
    idx = [i for i, p in enumerate(pairs['evaluation']) if p['theme'] == theme]
    themed[theme] = evaluation.score_pairs(labels['evaluation'][idx], distances['evaluation'][idx], threshold)
results = {
    'metrics': metrics, 'threshold': threshold, 'methods': method_results,
    'theme_metrics': themed,
    'robustness': {k: {'median': float(np.median(v)), 'min': int(min(v)), 'max': int(max(v))} for k, v in robustness.items()},
    'retrieval': {'query': query_id, 'results': [{'source_id': candidates[i]['source_id'], 'distance': int(d), 'same_source': candidates[i]['source_id'] == query_source} for i, d in top_results]},
    'pairs': {split: [{**p, 'distance': int(d), 'prediction': int(d <= threshold)} for p, d in zip(rows, distances[split], strict=True)] for split, rows in pairs.items()},
    'source_order': source_ids, 'demo_hash': ''.join(map(str, demo_hash)), 'demo_ll_shape': list(channel_steps['R'][1].shape),
    'config': config,
    'versions': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__, 'pywavelets': pywt.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
}
assets = root / 'docs/assets/lab04'
assets.mkdir(parents=True, exist_ok=True)
(assets / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
"""


def main() -> None:
    """Run a fresh kernel; persist the executed notebook without the audit cell."""
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks/labs/lab_04.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(AUDIT))
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    notebook.cells.pop()
    images = [
        output.data["image/png"]
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    if len(images) != len(FIGURES):
        raise ValueError(f"Expected {len(FIGURES)} figures; got {len(images)}.")
    nbformat.write(notebook, path)
    for name, data in zip(FIGURES, images, strict=True):
        (root / f"docs/assets/lab04/{name}.png").write_bytes(base64.b64decode(data))
    print(f"Lab04 execution, audit and {len(images)} figures: PASS")


if __name__ == "__main__":
    main()
