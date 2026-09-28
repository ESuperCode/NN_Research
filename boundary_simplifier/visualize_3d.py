"""
visualize_3d.py

Builds an interactive 3D view comparing:
  - the baseline network's decision surface
  - the simplified surrogate's decision surface
  - the actual train/test data points
inside a free-roaming "editor camera" (orbit / pan / zoom via mouse,
same interaction model as Blender/Unity's viewport camera), rendered with
Three.js in a self-contained HTML file.

IMPORTANT HONESTY NOTE
----------------------
Text data typically lives in a space with thousands of dimensions. To draw
anything in 3D, this script projects onto the top-3 principal components of
the training data (a *separate* 3-component PCA used only for plotting --
the surrogate model itself can use more components internally for
accuracy). The two surfaces you see are the 0.5-probability level sets of
each model evaluated on points reconstructed from this 3D subspace back
into the original feature space. That means:
  - This is an approximate 3D cross-section/projection of each model's true
    high-dimensional boundary, not the literal full-dimensional boundary.
  - Because the reconstruction from 3 components discards most of the
    variance in real text data, treat this as a qualitative comparison of
    the two surfaces' shape and relative position, not a precise picture
    of either model's full decision function.
"""

import json
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from skimage import measure


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


class _Projector:
    """
    Maps between the model's native input space and the 3D space used for
    plotting. Two modes:
      - native (identity): input is already 3D (e.g. the synthetic
        stress-test patterns). Grid points map back to feature space with
        zero information loss, so the extracted surface is the EXACT
        0.5-level boundary, not an approximation.
      - PCA: input is high-dimensional (e.g. TF-IDF text). Grid points are
        reconstructed via inverse PCA, which loses whatever variance isn't
        captured by 3 components -- the extracted surface is an
        approximate cross-section, not the literal full-dimensional
        boundary. See the module docstring.
    """

    def __init__(self, Z_train, pca=None, clip_non_negative=False):
        self.pca = pca
        self.clip_non_negative = clip_non_negative
        self.mins = Z_train.min(axis=0)
        self.maxs = Z_train.max(axis=0)
        self.ranges = np.maximum(self.maxs - self.mins, 1e-6)
        self.is_exact = pca is None

    def inverse_transform(self, Z):
        X = Z if self.pca is None else self.pca.inverse_transform(Z)
        if self.clip_non_negative:
            X = np.clip(X, 0, None)
        return X


def _evaluate_on_grid(model, projector, grid_res, margin=1.5):
    """Builds a 3D grid over the plotting space, maps it back into feature
    space via the projector, queries the model, and returns the resulting
    scalar field + grid geometry."""
    lo = projector.mins - margin * projector.ranges
    hi = projector.maxs + margin * projector.ranges

    axes = [np.linspace(lo[i], hi[i], grid_res) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    X_recon = projector.inverse_transform(grid_points)
    proba = model.predict_proba(X_recon)
    field = proba.reshape(grid_res, grid_res, grid_res)
    spacing = tuple((hi[i] - lo[i]) / (grid_res - 1) for i in range(3))
    return field, lo, spacing


def _mesh_to_json(field, origin, spacing, level=0.5):
    try:
        verts, faces, _, _ = measure.marching_cubes(field, level=level, spacing=spacing)
    except (ValueError, RuntimeError):
        return None  # no isosurface at this level within this 3D slice
    verts = verts + origin  # shift from grid-local to PCA coordinate space
    return {"vertices": verts.tolist(), "faces": faces.tolist()}


def build_visualization(
    baseline_model,
    simplified_model,
    X_train, y_train,
    X_test, y_test,
    out_path="/mnt/user-data/outputs/boundary_visualization.html",
    grid_res=28,
    native_3d=None,
    clip_non_negative=None,
):
    """
    native_3d : if True, treat X_train/X_test as already 3D and skip PCA
        entirely -- the extracted surfaces are then the EXACT boundary.
        If None (default), auto-detects: native when input dimensionality
        is already 3.
    clip_non_negative : whether to clip reconstructed points to >= 0
        (appropriate for TF-IDF text features, wrong for real-valued
        coordinates like the synthetic 3D patterns). If None, defaults to
        `not native_3d` (i.e. only clips in the PCA/text case).
    """
    X_train_dense = _to_dense(X_train)
    X_test_dense = _to_dense(X_test)

    if native_3d is None:
        native_3d = X_train_dense.shape[1] == 3
    if clip_non_negative is None:
        clip_non_negative = not native_3d

    if native_3d:
        assert X_train_dense.shape[1] == 3, "native_3d=True requires 3D input."
        pca = None
        Z_train, Z_test = X_train_dense, X_test_dense
        variance_note = [1.0, 1.0, 1.0]  # exact -- no information discarded
    else:
        pca = PCA(n_components=3, random_state=42)
        Z_train = pca.fit_transform(X_train_dense)
        Z_test = pca.transform(X_test_dense)
        variance_note = pca.explained_variance_ratio_.tolist()

    projector = _Projector(Z_train, pca=pca, clip_non_negative=clip_non_negative)

    baseline_field, origin, spacing = _evaluate_on_grid(baseline_model, projector, grid_res)
    simplified_field, _, _ = _evaluate_on_grid(simplified_model, projector, grid_res)

    baseline_mesh = _mesh_to_json(baseline_field, origin, spacing)
    simplified_mesh = _mesh_to_json(simplified_field, origin, spacing)

    payload = {
        "baseline_mesh": baseline_mesh,
        "simplified_mesh": simplified_mesh,
        "train_points": Z_train.tolist(),
        "train_labels": np.asarray(y_train).tolist(),
        "test_points": Z_test.tolist(),
        "test_labels": np.asarray(y_test).tolist(),
        "explained_variance_ratio": variance_note,
        "is_exact": bool(projector.is_exact),
    }

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote visualization to {out_path}")
    if native_3d:
        print("Native 3D input: extracted surfaces are the EXACT 0.5-level boundary.")
    else:
        print(f"3D PCA explained variance ratio (how much of the real structure "
              f"this view actually shows): {variance_note}")
    return out_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Decision Boundary: Baseline vs Simplified</title>
<style>
  html, body { margin:0; height:100%; background:#111318; color:#eee; font-family: system-ui, sans-serif; overflow:hidden; }
  #panel { position:absolute; top:12px; left:12px; z-index:10; background:rgba(20,22,28,0.85);
           padding:14px 16px; border-radius:10px; font-size:13px; line-height:1.5; max-width:320px; }
  #panel h1 { font-size:15px; margin:0 0 8px 0; }
  #panel label { display:flex; align-items:center; gap:8px; margin:4px 0; cursor:pointer; }
  #panel .swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
  #panel .note { opacity:0.7; margin-top:10px; font-size:11.5px; }
  #hint { position:absolute; bottom:10px; left:12px; z-index:10; font-size:11.5px; opacity:0.6; }
</style>
</head>
<body>
<div id="panel">
  <h1>Decision boundary comparison</h1>
  <label><input type="checkbox" id="toggle-baseline" checked><span class="swatch" style="background:#3aa0ff"></span> Baseline network surface</label>
  <label><input type="checkbox" id="toggle-simplified" checked><span class="swatch" style="background:#ff7a3a"></span> Simplified surrogate surface</label>
  <hr style="border-color:#333; margin:8px 0;">
  <div style="opacity:0.75; font-size:11.5px; margin-bottom:2px;">TRAIN (small, translucent dots)</div>
  <label><input type="checkbox" id="toggle-train-c0" checked><span class="swatch" style="background:#27ae60"></span> Train — Class 0</label>
  <label><input type="checkbox" id="toggle-train-c1" checked><span class="swatch" style="background:#e74c3c"></span> Train — Class 1</label>
  <div style="opacity:0.75; font-size:11.5px; margin:6px 0 2px 0;">TEST (large, solid dots)</div>
  <label><input type="checkbox" id="toggle-test-c0" checked><span class="swatch" style="background:#00bcd4"></span> Test — Class 0</label>
  <label><input type="checkbox" id="toggle-test-c1" checked><span class="swatch" style="background:#ffd54f"></span> Test — Class 1</label>
  <div class="note" id="variance-note"></div>
</div>
<div id="hint">drag = orbit &nbsp;·&nbsp; right-drag / two-finger = pan &nbsp;·&nbsp; scroll = zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA_JSON__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111318);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(5,8,6);
scene.add(dl);

// Editor-style camera: left-drag orbit, right-drag pan, scroll zoom.
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;

function buildMesh(meshData, color, opacity) {
  if (!meshData) return null;
  const geo = new THREE.BufferGeometry();
  const verts = new Float32Array(meshData.vertices.flat());
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(meshData.faces.flat());
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color, opacity, transparent:true, side: THREE.DoubleSide, roughness:0.5, metalness:0.05
  });
  return new THREE.Mesh(geo, mat);
}

const baselineMesh = buildMesh(DATA.baseline_mesh, 0x3aa0ff, 0.35);
const simplifiedMesh = buildMesh(DATA.simplified_mesh, 0xff7a3a, 0.35);
if (baselineMesh) scene.add(baselineMesh);
if (simplifiedMesh) scene.add(simplifiedMesh);

// Four genuinely distinct groups: {train, test} x {class 0, class 1}.
// Class controls COLOR (consistent between train/test so you can compare
// the same class across splits). Train vs test controls SIZE + OPACITY
// (train = small/translucent "cloud", test = large/solid "markers") so
// the split is visible independently of color for anyone who is
// colorblind or just wants a second visual cue.
function splitByLabel(points, labels) {
  const c0 = [], c1 = [];
  for (let i = 0; i < points.length; i++) {
    (labels[i] === 1 ? c1 : c0).push(points[i]);
  }
  return [c0, c1];
}

function makeCloud(pts, color, size, opacity) {
  if (pts.length === 0) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts.flat()), 3));
  const m = new THREE.PointsMaterial({color, size, opacity, transparent: opacity < 1,
                                       sizeAttenuation: true});
  return new THREE.Points(g, m);
}

const [trainC0pts, trainC1pts] = splitByLabel(DATA.train_points, DATA.train_labels);
const [testC0pts, testC1pts] = splitByLabel(DATA.test_points, DATA.test_labels);

const COLOR_CLASS0 = 0x27ae60; // green
const COLOR_CLASS1 = 0xe74c3c; // red
const COLOR_TEST_CLASS0 = 0x00bcd4; // cyan  (kept a distinct shade from train's class 0
const COLOR_TEST_CLASS1 = 0xffd54f; // gold  (so train/test are distinguishable by color too)

const trainClass0 = makeCloud(trainC0pts, COLOR_CLASS0, 0.055, 0.55);
const trainClass1 = makeCloud(trainC1pts, COLOR_CLASS1, 0.055, 0.55);
const testClass0 = makeCloud(testC0pts, COLOR_TEST_CLASS0, 0.09, 1.0);
const testClass1 = makeCloud(testC1pts, COLOR_TEST_CLASS1, 0.09, 1.0);

[trainClass0, trainClass1, testClass0, testClass1].forEach(p => { if (p) scene.add(p); });

// Frame the camera around the data extents.
const allPts = DATA.train_points.concat(DATA.test_points);
const box = new THREE.Box3();
allPts.forEach(p => box.expandByPoint(new THREE.Vector3(p[0],p[1],p[2])));
const center = box.getCenter(new THREE.Vector3());
const size = box.getSize(new THREE.Vector3()).length();
camera.position.copy(center).add(new THREE.Vector3(size*0.7, size*0.5, size*0.7));
controls.target.copy(center);
controls.update();

document.getElementById('toggle-baseline').addEventListener('change', e => { if (baselineMesh) baselineMesh.visible = e.target.checked; });
document.getElementById('toggle-simplified').addEventListener('change', e => { if (simplifiedMesh) simplifiedMesh.visible = e.target.checked; });
document.getElementById('toggle-train-c0').addEventListener('change', e => { if (trainClass0) trainClass0.visible = e.target.checked; });
document.getElementById('toggle-train-c1').addEventListener('change', e => { if (trainClass1) trainClass1.visible = e.target.checked; });
document.getElementById('toggle-test-c0').addEventListener('change', e => { if (testClass0) testClass0.visible = e.target.checked; });
document.getElementById('toggle-test-c1').addEventListener('change', e => { if (testClass1) testClass1.visible = e.target.checked; });

if (DATA.is_exact) {
  document.getElementById('variance-note').textContent =
    'Input is natively 3D, so these are the EXACT 0.5-probability decision surfaces of each model -- no projection or approximation.';
} else {
  const vr = DATA.explained_variance_ratio.map(v => (v*100).toFixed(1)+'%').join(' / ');
  document.getElementById('variance-note').textContent =
    'This is a 3-component PCA projection (explains ' + vr + ' of variance per axis). ' +
    'Surfaces are approximate 3D cross-sections of each model\\'s true high-dimensional boundary, not the exact boundary itself.';
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); }
animate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    from data_utils import load_text_classification_data
    from baseline_model import train_baseline
    from boundary_simplifier import convert_to_simplified_boundary

    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    baseline = train_baseline(X_train, y_train)
    simplified = convert_to_simplified_boundary(baseline, X_train, n_components=10)

    build_visualization(
        baseline, simplified,
        X_train, y_train, X_test, y_test,
        out_path="/home/claude/boundary_project/boundary_visualization.html",
    )
