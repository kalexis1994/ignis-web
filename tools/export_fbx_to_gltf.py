import sys
from pathlib import Path

import bpy


def parse_args():
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender --background --python export_fbx_to_gltf.py -- <fbx_path> <out_dir> [texture_dir]")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit("Expected <fbx_path> <out_dir> [texture_dir]")
    fbx_path = Path(args[0]).resolve()
    out_dir = Path(args[1]).resolve()
    texture_dir = Path(args[2]).resolve() if len(args) >= 3 else None
    return fbx_path, out_dir, texture_dir


def count_scene():
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    material_count = len(bpy.data.materials)
    image_count = len(bpy.data.images)
    tri_count = 0
    vert_count = 0
    for obj in mesh_objects:
        mesh = obj.data
        tri_count += sum(max(len(poly.vertices) - 2, 0) for poly in mesh.polygons)
        vert_count += len(mesh.vertices)
    return {
        "mesh_objects": len(mesh_objects),
        "materials": material_count,
        "images": image_count,
        "triangles": tri_count,
        "vertices": vert_count,
    }


def print_stats(prefix):
    stats = count_scene()
    print(
        f"{prefix}: "
        f"objects={stats['mesh_objects']} "
        f"materials={stats['materials']} "
        f"images={stats['images']} "
        f"verts={stats['vertices']} "
        f"tris={stats['triangles']}"
    )


def main():
    fbx_path, out_dir, texture_dir = parse_args()
    if not fbx_path.is_file():
        raise SystemExit(f"FBX not found: {fbx_path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    print(f"Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_image_search=True)

    if texture_dir and texture_dir.is_dir():
      try:
        bpy.ops.file.find_missing_files(directory=str(texture_dir))
      except Exception as exc:
        print(f"find_missing_files skipped: {exc}")

    print_stats("Imported scene")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene.gltf"

    print(f"Exporting glTF: {out_path}")
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLTF_SEPARATE",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_yup=True,
        export_apply=False,
        use_selection=False,
        check_existing=False,
    )

    print(f"Export complete: {out_path}")


if __name__ == "__main__":
    main()
