import json
import math
import os
import sys
from pathlib import Path

import bpy

def parse_args() -> dict:
    args = sys.argv
    if "--" not in args:
        raise RuntimeError("No configuration payload passed to Blender script.")

    raw_args = args[args.index("--") + 1:]
    config_path = None

    for i, arg in enumerate(raw_args):
        if arg == "--config" and i + 1 < len(raw_args):
            config_path = raw_args[i + 1]
            break

    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(confirm=False)

    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for item in block:
            if item.users == 0:
                block.remove(item)


def setup_world_lighting(theme: str) -> None:
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("FalconAI_World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.05, 0.08, 0.15, 1.0)
        bg_node.inputs["Strength"].default_value = 1.2

    key_light_data = bpy.data.lights.new(name="Key_Light", type="SUN")
    key_light_data.energy = 3.5
    key_light_data.color = (1.0, 0.95, 0.85)
    key_light_obj = bpy.data.objects.new(name="Key_Light", object_data=key_light_data)
    bpy.context.collection.objects.link(key_light_obj)
    key_light_obj.location = (5.0, -5.0, 8.0)
    key_light_obj.rotation_euler = (math.radians(45), 0, math.radians(30))

    fill_light_data = bpy.data.lights.new(name="Fill_Light", type="SUN")
    fill_light_data.energy = 1.5
    fill_light_data.color = (0.7, 0.85, 1.0)
    fill_light_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light_obj)
    fill_light_obj.location = (-5.0, -3.0, 5.0)

def create_character_mesh(face_texture_path: str = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0, 0, 1.5))
    character = bpy.context.active_object
    character.name = "Hero_Character"

    subsurf = character.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.render_levels = 2
    subsurf.levels = 1

    mat = bpy.data.materials.new(name="Character_Material")
    mat.use_nodes = True
    character.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    if face_texture_path and os.path.exists(face_texture_path):
        try:
            tex_image = nodes.new("ShaderNodeTexImage")
            tex_image.image = bpy.data.images.load(face_texture_path)
            mat.node_tree.links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception as e:
            print(f"[Blender Internal Warning] Failed to map face texture: {e}")
            bsdf.inputs["Base Color"].default_value = (0.9, 0.75, 0.65, 1.0)
    else:
        bsdf.inputs["Base Color"].default_value = (0.9, 0.75, 0.65, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.4
    return character

def setup_animated_camera(total_frames: int) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name="Cinematic_Camera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new(name="Cinematic_Camera", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = (0.0, -4.5, 1.8)
    cam_obj.rotation_euler = (math.radians(85), 0, 0)
    cam_obj.keyframe_insert(data_path="location", frame=1)

    cam_obj.location = (0.0, -3.8, 1.6)
    cam_obj.keyframe_insert(data_path="location", frame=total_frames)

    return cam_obj

def configure_render_engine(settings: dict) -> None:
    scene = bpy.context.scene
    engine_type = settings.get("engine", "EEVEE").upper()

    if engine_type == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = settings.get("samples", 64)
        if settings.get("use_gpu", True):
            scene.cycles.device = "GPU"
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = True
    else:
        scene.render.engine = "BLENDER_EEVEE" if hasattr(bpy.types, "RenderSettings") else "EEVEE"
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = settings.get("samples", 32)
            scene.eevee.use_bloom = True
            scene.eevee.use_gtao = True

    scene.render.resolution_x = settings.get("resolution_width", 1920)
    scene.render.resolution_y = settings.get("resolution_height", 1080)
    scene.render.resolution_percentage = 100
    scene.render.fps = settings.get("fps", 24)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

def main() -> None:
    payload = parse_args()
    render_settings = payload.get("render_settings", {})
    output_dir = Path(render_settings.get("output_dir", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()

    setup_world_lighting(theme=payload.get("theme", "magical"))
    character = create_character_mesh(face_texture_path=payload.get("face_texture_path"))

    fps = render_settings.get("fps", 24)
    duration_seconds = 5
    total_frames = fps * duration_seconds
    skip_frames = set(payload.get("resume_skip_frames", []))

    print(f"[TOTAL_FRAMES]:{total_frames}", flush=True)

    setup_animated_camera(total_frames=total_frames)

    configure_render_engine(render_settings)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames

    for frame in range(1, total_frames + 1):
        if frame in skip_frames:
            print(f"[FRAME_SKIPPED]:{frame}", flush=True)
            continue

        scene.frame_set(frame)

        character.rotation_euler.z = math.radians(math.sin(frame * 0.05) * 15)
        character.keyframe_insert(data_path="rotation_euler", frame=frame)

        frame_filename = f"frame_{frame:04d}.png"
        scene.render.filepath = str(output_dir / frame_filename)

        bpy.ops.render.render(write_still=True)
        print(f"[FRAME_RENDERED]:{frame}", flush=True)

    print("[BLENDER_RENDER_COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
