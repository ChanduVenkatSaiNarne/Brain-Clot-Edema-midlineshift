"""Turn a 3D label map into a coloured 3D model (.glb) for the web/VR viewer, 3D Slicer, Blender or Unity.

  python export_mesh.py --labels pred/brats0.nii.gz --preset brats  --context BraTS2021_00000_flair.nii.gz --out brats0.glb
  python export_mesh.py --labels pred/isles1.nii.gz --preset stroke --context dwi.nii.gz                 --out isles1.glb
  python export_mesh.py --labels pred/ct1.nii.gz    --preset ich    --context ct.nii.gz --context-mode ct-skull --out ct1.glb

How: each label -> light Gaussian smoothing -> marching cubes -> triangle mesh in millimetres
(scanner coordinates from the NIfTI affine) -> converted to metres, Y-up (glTF convention).
The optional context image adds a see-through brain (MRI) or skull (CT) surface for orientation.
Also writes <out>.json with each structure's volume in mL.
"""

import argparse
import json
import struct

import numpy as np
from scipy import ndimage
from skimage import measure

PRESETS = {   # value: (name, RGBA)
    "brats": {1: ("Edema", (0.95, 0.85, 0.20, 0.35)), 2: ("Necrotic core", (0.55, 0.20, 0.75, 0.9)),
              3: ("Enhancing tumour", (0.90, 0.15, 0.15, 1.0))},
    "stroke": {1: ("Infarct", (0.10, 0.65, 1.00, 0.95))},
    "ich": {1: ("Haemorrhage", (0.85, 0.05, 0.05, 1.0))},
    "bhsd": {i: (f"Bleed type {i}", c) for i, c in enumerate(
        [(0.85, 0.05, 0.05, 1), (1.0, 0.55, 0.0, 1), (0.95, 0.85, 0.1, 1), (0.2, 0.75, 0.3, 1), (0.6, 0.3, 0.85, 1)], 1)},
}
CONTEXT_RGBA = (0.85, 0.85, 0.88, 0.12)


# --------------------------------------------------------------------------- geometry
def mask_to_mesh(mask, affine, sigma=1.0, step=1):
    """Binary 3D mask -> (vertices in mm, triangle indices). Returns None if the mask is empty."""
    if mask.sum() == 0:
        return None
    vol = ndimage.gaussian_filter(np.pad(mask.astype(np.float32), 2), sigma) if sigma > 0 else np.pad(mask.astype(np.float32), 2)
    if vol.max() <= 0.5:
        vol = np.pad(mask.astype(np.float32), 2)          # tiny lesion: skip smoothing so it does not vanish
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, step_size=step)
    verts -= 2                                            # undo padding
    mm = verts @ affine[:3, :3].T + affine[:3, 3]
    # make triangles face outwards (positive enclosed volume), whatever the scanner's voxel order
    t = mm[faces]
    if np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return mm, np.ascontiguousarray(faces, dtype=np.uint32)


def to_gltf_space(mm, center):
    """Scanner RAS millimetres -> glTF metres (x right, y up, z towards the viewer)."""
    p = (mm - center) / 1000.0
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1).astype(np.float32)


def vertex_normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------- minimal GLB writer
def write_glb(path, meshes):
    """meshes: list of dicts {name, verts (N,3) float32, faces (M,3) uint32, rgba}."""
    gltf = {"asset": {"version": "2.0", "generator": "Brain-Clot-Edema-midlineshift export_mesh.py"},
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
            "buffers": [], "bufferViews": [], "accessors": []}
    blob = bytearray()

    def add_view(data, target):
        while len(blob) % 4:
            blob.append(0)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(gltf["bufferViews"]) - 1

    for m in meshes:
        v, f = m["verts"], m["faces"]
        nrm = vertex_normals(v, f)
        acc = len(gltf["accessors"])
        gltf["accessors"] += [
            {"bufferView": add_view(v.tobytes(), 34962), "componentType": 5126, "count": len(v), "type": "VEC3",
             "min": v.min(0).tolist(), "max": v.max(0).tolist()},
            {"bufferView": add_view(nrm.tobytes(), 34962), "componentType": 5126, "count": len(v), "type": "VEC3"},
            {"bufferView": add_view(f.astype(np.uint32).tobytes(), 34963), "componentType": 5125,
             "count": int(f.size), "type": "SCALAR"},
        ]
        r, g, b, a = m["rgba"]
        mat = {"name": m["name"], "doubleSided": True,
               "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, a], "metallicFactor": 0.0, "roughnessFactor": 0.55}}
        if a < 1:
            mat["alphaMode"] = "BLEND"
        gltf["materials"].append(mat)
        gltf["meshes"].append({"name": m["name"], "primitives": [
            {"attributes": {"POSITION": acc, "NORMAL": acc + 1}, "indices": acc + 2,
             "material": len(gltf["materials"]) - 1}]})
        gltf["nodes"].append({"name": m["name"], "mesh": len(gltf["meshes"]) - 1})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"].append({"byteLength": len(blob)})
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def read_glb(path):
    """Parse a GLB back (used by the self-test). Returns the JSON part and the binary length."""
    data = open(path, "rb").read()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    jlen, jtype = struct.unpack("<II", data[12:20])
    assert jtype == 0x4E4F534A
    gltf = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack("<II", data[20 + jlen:28 + jlen])
    assert btype == 0x004E4942 and blen == gltf["buffers"][0]["byteLength"]
    return gltf, blen


# --------------------------------------------------------------------------- main
def build_meshes(lab, affine, preset, context=None, context_affine=None, context_mode="mri", sigma=1.0):
    meshes, volumes, pts = [], {}, []
    vox_ml = abs(np.linalg.det(affine[:3, :3])) / 1000.0
    raw = []
    for value, (name, rgba) in preset.items():
        mask = lab == value
        volumes[name] = round(float(mask.sum() * vox_ml), 3)
        mesh = mask_to_mesh(mask, affine, sigma)
        if mesh:
            raw.append((name, rgba, *mesh))
            pts.append(mesh[0])
    if context is not None:
        if context_mode == "ct-skull":
            cmask = context > 300                                    # bone, in HU
        else:
            nz = context[context > 0]                                # skull-stripped MRI: background is 0
            cmask = context > (0.3 * np.percentile(nz, 5) if nz.size else 0)
            cmask = ndimage.binary_fill_holes(ndimage.binary_opening(cmask, iterations=2))
        cmesh = mask_to_mesh(cmask, context_affine, sigma=1.5, step=2)
        if cmesh:
            raw.insert(0, ("Brain" if context_mode == "mri" else "Skull", CONTEXT_RGBA, *cmesh))
            pts = [cmesh[0]]                                         # centre on the head
    if not raw:
        raise SystemExit("Nothing to export: label map is empty for this preset")
    center = np.concatenate(pts).mean(0)
    for name, rgba, mm, faces in raw:
        meshes.append({"name": name, "rgba": rgba, "verts": to_gltf_space(mm, center), "faces": faces})
    return meshes, volumes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Label map .nii.gz (prediction or ground truth)")
    ap.add_argument("--preset", default="brats", choices=list(PRESETS))
    ap.add_argument("--context", help="MRI/CT .nii.gz for a see-through brain or skull surface")
    ap.add_argument("--context-mode", default="mri", choices=["mri", "ct-skull"])
    ap.add_argument("--sigma", type=float, default=1.0, help="Surface smoothing (voxels); 0 = none")
    ap.add_argument("--out", required=True, help="Output .glb")
    a = ap.parse_args()

    import nibabel as nib
    li = nib.load(a.labels)
    lab = np.rint(np.asanyarray(li.dataobj)).astype(np.int16)
    ctx = ctx_aff = None
    if a.context:
        ci = nib.load(a.context)
        ctx, ctx_aff = np.asanyarray(ci.dataobj).astype(np.float32), ci.affine
        if ctx.ndim == 4:
            ctx = ctx[..., 0]
    meshes, volumes = build_meshes(lab, li.affine, PRESETS[a.preset], ctx, ctx_aff, a.context_mode, a.sigma)
    write_glb(a.out, meshes)
    json.dump({"volumes_ml": volumes, "meshes": [m["name"] for m in meshes]},
              open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=2)
    tri = sum(len(m["faces"]) for m in meshes)
    print(f"Saved {a.out}: {len(meshes)} meshes, {tri:,} triangles | volumes (mL) {volumes}")


if __name__ == "__main__":
    main()
