import bpy
import csv
from math import radians
from mathutils import Vector

csv_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/IMG1.csv'
obj_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/mesh_folder/ModelExport.obj'
background_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/IMG1.png'
output_image_path = "/Users/sudeepsharma/Desktop/blender_positioning/render_output.png"
output_blend_path = '/Users/sudeepsharma/Desktop/blender_positioning/blend.blend'


def read_csv(csv_path):
    data = {}
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            key = row[0].strip()
            values = list(map(float, row[1:]))
            data[key] = values
    return data

def setup_camera(camera_pos, camera_rot, focal_length, sensor_size):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.location = Vector(camera_pos)
    camera.rotation_euler = [radians(r) for r in camera_rot]
    camera.data.lens = focal_length
    camera.data.sensor_width, camera.data.sensor_height = sensor_size
    bpy.context.scene.camera = camera

def load_obj(obj_path, position, rotation):
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.location = Vector(position)
    # Example custom rotation (adjust as needed):
    obj.rotation_euler = [0,0, 0]


def setup_scene(obj_path, bg_path, csv_path, blend_path, render_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    data = read_csv(csv_path)
    load_obj(obj_path, data["model position"], data["model rotation"])
    setup_camera(data["camera position"], data["camera rotation"], 6.99, (9.99, 7.495))

    scene = bpy.context.scene
    scene.render.film_transparent = True  # Key setting to enable transparency

    scene.use_nodes = True
    tree = scene.node_tree

    # Clear existing compositor nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    bg_image_node = tree.nodes.new(type='CompositorNodeImage')
    bg_image_node.image = bpy.data.images.load(bg_path)
    alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
    composite = tree.nodes.new(type='CompositorNodeComposite')

    tree.links.new(bg_image_node.outputs['Image'], alpha_over.inputs[1])
    tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
    tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])

    scene.render.filepath = render_path
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=blend_path)


# Run the setup
setup_scene(
    obj_path=obj_path,
    bg_path=background_path,
    csv_path=csv_path,
    blend_path=output_blend_path,
    render_path=output_image_path
)
