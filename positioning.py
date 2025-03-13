import bpy
import csv
import numpy as np
import os
import mathutils
from mathutils import Matrix, Vector
import traceback


# Path configuration
image_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/image_target.png'
obj_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/mesh_folder/model.obj'
csv_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/IMG1.csv'
background_path = '/Users/sudeepsharma/Desktop/blender_positioning/src/IMG1.png'
output_image_path = "/Users/sudeepsharma/Desktop/blender_positioning/render_output.png"
output_blend_path = '/Users/sudeepsharma/Desktop/blender_positioning/blend.blend'

def import_obj_at_origin(obj_path, scale_factor=1.0):
    """
    Import an OBJ file and place it at the world origin (0,0,0).
    """
    if not os.path.exists(obj_path):
        error_message = f"ERROR: OBJ file does not exist at path: {obj_path}"
        print(error_message)
        return None

    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    print("import obj called")
    try:
        # Use the correct import operator for Blender 4.3
        bpy.ops.wm.obj_import(filepath=obj_path)
        print(f"Imported OBJ: {obj_path}")

        imported_objects = [obj for obj in bpy.context.selected_objects]
        if not imported_objects:
            warning_message = "WARNING: No objects were imported or selected"
            print(warning_message)
            return None

        # Create a parent empty object
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        parent = bpy.context.object
        parent.name = "OBJ_Parent"

        # Parent all imported objects
        for obj in imported_objects:
            obj.parent = parent
            obj.matrix_parent_inverse = parent.matrix_world.inverted()

        # Apply scale
        parent.scale = Vector((scale_factor, scale_factor, scale_factor))

        # Add coordinate axes
        bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
        axes = bpy.context.object
        axes.scale = Vector((0.2, 0.2, 0.2))
        axes.name = "OBJAxes"

        return parent

    except Exception as e:
        error_message = f"Error importing OBJ: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        return None
     
def read_matrix_csv(csv_path):
    """Read MVP matrices from CSV file exported from ARKit"""
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get header
        
        # Initialize matrices and additional data
        model_matrix = None
        view_matrix = None
        projection_matrix = None
        screen_aspect_ratio = None
        
        for row in reader:
            if not row:  # Skip empty rows
                continue
                
            matrix_type = row[0].strip()
            
            if matrix_type == "screenAspectratio" and len(row) > 1:
                try:
                    screen_aspect_ratio = float(row[1])
                    print(f"Screen aspect ratio: {screen_aspect_ratio}")
                    continue
                except ValueError:
                    print(f"Warning: Could not parse aspect ratio {row[1]} as float")
            
            if len(row) < 17:  # Need at least matrix_type + 16 values for matrices
                print(f"Warning: Row has insufficient values: {row}")
                continue
            
            # Parse the 16 matrix values
            values = []
            for i in range(1, 17):
                try:
                    values.append(float(row[i]))
                except ValueError:
                    print(f"Warning: Could not parse value {row[i]} as float")
                    values.append(0.0)
            
            # Reshape into 4x4 matrix (column-major order from ARKit)
            matrix_values = np.array(values).reshape(4, 4)
            
            if matrix_type == "modelMatrix":
                model_matrix = matrix_values
                print("Parsed model matrix")
            elif matrix_type == "cameraTransform":
                view_matrix = matrix_values
                print("Parsed camera transform matrix")
            elif matrix_type == "projectionMatrix":
                projection_matrix = matrix_values
                print("Parsed projection matrix")
    
    if model_matrix is None or view_matrix is None or projection_matrix is None:
        raise ValueError("One or more required matrices missing from CSV")
        
    # Convert numpy arrays to Blender matrices
    model_matrix_bl = Matrix(model_matrix.tolist())
    view_matrix_bl = Matrix(view_matrix.tolist())
    projection_matrix_bl = Matrix(projection_matrix.tolist())
    
    # Display matrices for debugging
    print("\nModel Matrix (from CSV):")
    print(model_matrix_bl)
    
    print("\nView Matrix (from CSV):")
    print(view_matrix_bl)
    
    print("\nProjection Matrix (from CSV):")
    print(projection_matrix_bl)
    
    return {
        "model_matrix": model_matrix_bl,
        "view_matrix": view_matrix_bl,
        "projection_matrix": projection_matrix_bl,
        "screen_aspect_ratio": screen_aspect_ratio
    }


def fix_model_matrix(model_matrix):
    """Apply corrections to ARKit model matrix for Blender compatibility"""
    # Create a correction matrix for Blender
    # Blender uses right-handed Z-up, ARKit uses right-handed Y-up
    correction = Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1)
    ))
    
    # Apply correction
    fixed_matrix = correction @ model_matrix @ correction.inverted()
    
    # For persistent alignment issues, you can add specific rotations here
#    rotation_fix = Matrix.Rotation(np.radians(180), 4, 'Z')
#    fixed_matrix = fixed_matrix 
    
    print("\nCorrected Model Matrix:")
    print(fixed_matrix)
    
    return fixed_matrix

def fix_camera_matrix(view_matrix):
    """Convert ARKit camera view matrix to Blender camera matrix"""
    # Invert view matrix to get camera-to-world transform
    camera_matrix = view_matrix.inverted()
    
    # Apply coordinate system correction
    correction = Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1)
    ))
    
    fixed_camera = correction @ camera_matrix
    
    # For persistent camera alignment issues, add specific adjustments
    # Example: rotate camera 180 degrees around Z axis
#    camera_rotation_fix = Matrix.Rotation(np.radians(180), 4, 'Z')
#    fixed_camera = fixed_camera
    
#    local_y_rotation = Matrix.Rotation(np.radians(90), 4, 'Y')
#    fixed_camera = fixed_camera @ local_y_rotation
    
    print("\nCorrected Camera Matrix:")
    print(fixed_camera)
    
    return fixed_camera

def setup_camera_from_matrices(view_matrix, projection_matrix):
    """Setup camera using view and projection matrices from ARKit"""
    # Create new camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    
    # Fix camera matrix for Blender
    camera_matrix = fix_camera_matrix(view_matrix)
    
    # Set camera position and rotation
    camera.matrix_world = camera_matrix
    
    # Set camera projection parameters
    camera_data = camera.data
    
    # Use projection matrix to set camera parameters
    # Extract field of view and aspect ratio
    if abs(projection_matrix[1][1]) > 0.0001:  # Avoid division by zero
        aspect_ratio = abs(projection_matrix[0][0] / projection_matrix[1][1])
        tan_half_fov = 1.0 / abs(projection_matrix[1][1])
        fov = 2 * np.arctan(tan_half_fov)
        
        # Convert FOV to focal length
        focal_length = camera_data.sensor_height / (2 * np.tan(fov/2))
        
        # Set camera parameters
        camera_data.lens_unit = 'MILLIMETERS'
        camera_data.lens = focal_length
        camera_data.sensor_width = camera_data.sensor_height * aspect_ratio
    else:
        print("Warning: Could not extract FOV from projection matrix, using defaults")
    
    # Set reasonable clipping values
    camera_data.clip_start = 0.001
    camera_data.clip_end = 1000.0
    
    # Make this the active camera
    bpy.context.scene.camera = camera
    
    # Set camera to portrait orientation
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = scene.render.resolution_y, scene.render.resolution_x
    
    # Print camera settings for debugging
    print(f"\nCamera settings (Portrait):")
    print(f"Focal length: {camera_data.lens}mm")
    print(f"Sensor size: {camera_data.sensor_width}x{camera_data.sensor_height}mm")
    print(f"Position: {camera.location}")
    print(f"Rotation: {camera.rotation_euler}")
    print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y} (Portrait)")
    
    return camera

def create_plane_with_matrix(model_matrix, image_path, size=1.0):
    """Create a plane, apply the model matrix, and set an image on the plane"""
    # Create a plane
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD')
    plane = bpy.context.active_object
    print(f"Created plane: {plane.name}")
    
    # Scale the plane by 2 in the Y direction
    plane.scale[1] = 2.0
    print(f"Scaled plane in Y direction by factor of 2")
    
    # Apply the scale to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Get original dimensions before applying transforms
    original_dimensions = plane.dimensions.copy()
    print(f"Original dimensions after Y scaling: {original_dimensions}")
    
    # Create a new material
    mat = bpy.data.materials.new(name="PlaneMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add a Principled BSDF shader
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Add an Image Texture node
    tex_image = nodes.new(type='ShaderNodeTexImage')
    tex_image.location = (-300, 0)
    
    # Load the image
    try:
        image = bpy.data.images.load(image_path)
        tex_image.image = image
        print(f"Loaded image: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Add an Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    # Link nodes
    links = mat.node_tree.links
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign the material to the plane
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)
    
    # Fix model matrix for ARKit to Blender conversion
    fixed_matrix = fix_model_matrix(model_matrix)
    
    # Apply the fixed matrix
    plane.matrix_world = fixed_matrix
    
    # Print transformed dimensions
    print(f"Transformed dimensions: {plane.dimensions}")
    
    # Add coordinate axes to visualize the model's orientation
    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    axes = bpy.context.object
    axes.matrix_world = fixed_matrix
    axes.scale = Vector((0.2, 0.2, 0.2))
    axes.name = "PlaneAxes"
    
    return plane

def setup_scene(bg_path, csv_path, blend_path, render_path, plane_size=1.0):
    """Setup the entire scene using ARKit matrices"""
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Read matrices from CSV
    try:
        matrices = read_matrix_csv(csv_path)
    except Exception as e:
        print(f"Error reading matrices: {e}")
        return
    
    # Create world origin indicator
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    origin = bpy.context.object
    origin.scale = Vector((0.2, 0.2, 0.2))
    origin.name = "WorldOrigin"
    
    # Create plane with model matrix
    try:
        obj = import_obj_at_origin(obj_path, scale_factor=1.0)
        plane = create_plane_with_matrix(matrices["model_matrix"], image_path, size=plane_size)
    except Exception as e:
        print(f"Error creating plane: {e}")
        return
    
    # Setup camera
    try:
        camera = setup_camera_from_matrices(matrices["view_matrix"], matrices["projection_matrix"])
    except Exception as e:
        print(f"Error setting up camera: {e}")
        return
    
    # Load and set specific background image dimensions
    try:
        bg_image = bpy.data.images.load(bg_path)
        aspect_ratio = matrices["screen_aspect_ratio"]
        scene = bpy.context.scene
        
        base_width = 1080
        height = int(base_width / aspect_ratio)
        scene.render.resolution_x = base_width
        scene.render.resolution_y = height
        
        scene.render.resolution_percentage = 100
    except Exception as e:
        print(f"Error loading background image: {e}")
    
    # Setup compositing nodes
    try:
        scene = bpy.context.scene
        scene.render.film_transparent = True
        scene.use_nodes = True
        tree = scene.node_tree
        
        # Clear existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)
        
        # Add nodes
        render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        bg_image_node = tree.nodes.new(type='CompositorNodeImage')
        bg_image_node.image = bg_image
        
        # For portrait mode (using fixed resolution)
        # Check if we need to rotate the background image
        if bg_image.size[0] > bg_image.size[1]:  # If background is landscape
            # Add rotation node to rotate the background image for portrait mode
            rotate_node = tree.nodes.new(type='CompositorNodeRotate')
            rotate_node.filter_type = 'BILINEAR'
            rotate_node.inputs[2].default_value = 1.5708  # 90 degrees in radians
            
            # Add scale node to ensure the background fits the resolution
            scale_node = tree.nodes.new(type='CompositorNodeScale')
            scale_node.space = 'RENDER_SIZE'
            
            # Link background image to rotation node, then to scale node
            tree.links.new(bg_image_node.outputs['Image'], rotate_node.inputs[0])
            tree.links.new(rotate_node.outputs['Image'], scale_node.inputs[0])
            
            # Alpha over with rotated and scaled background
            alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
            tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
            tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
        else:
            # Just scale the image to fit the render size
            scale_node = tree.nodes.new(type='CompositorNodeScale')
            scale_node.space = 'RENDER_SIZE'
            
            tree.links.new(bg_image_node.outputs['Image'], scale_node.inputs[0])
            
            # Alpha over with scaled background
            alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
            tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
            tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
        
        # Connect to composite
        composite = tree.nodes.new(type='CompositorNodeComposite')
        tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])
        
        print("Compositing setup complete for portrait mode with resolution 1179x2556")
    except Exception as e:
        print(f"Error setting up compositing: {e}")
    
    # # Setup lighting (add a simple environment light)
    # world = bpy.context.scene.world
    # world.use_nodes = True
    # bg_node = world.node_tree.nodes.get("Background")
    # if bg_node:
    #     bg_node.inputs["Color"].default_value = (1, 1, 1, 1)  # White light
    #     bg_node.inputs["Strength"].default_value = 1.0  # Moderate strength
    
    # # Setup render settings
    # scene.render.filepath = render_path
    # scene.render.image_settings.file_format = 'PNG'
    
    # # Use Cycles for better quality
    # scene.render.engine = 'CYCLES'
    # if scene.render.engine == 'CYCLES':
    #     scene.cycles.samples = 32  # Lower for faster preview, increase for final render
    
    # # Render and save
    # try:
    #     bpy.ops.render.render(write_still=True)
    #     print(f"Rendered image saved to: {render_path}")
    # except Exception as e:
    #     print(f"Error rendering: {e}")
    
    # Save .blend file
    try:
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"Blender file saved to: {blend_path}")
    except Exception as e:
        print(f"Error saving blend file: {e}")

def test_transforms():
    """Test function to verify matrix transformations"""
    print("\n=== Testing Matrix Transformations ===")
    
    # Test ARKit to Blender coordinate conversion
    test_point = Vector((1.0, 2.0, 3.0, 1.0))
    print(f"Original point: {test_point[:3]}")
    
    # Apply ARKit -> Blender transformation
    correction = Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1)
    ))
    
    converted_point = correction @ test_point
    print(f"Converted point: {converted_point[:3]}")
    
    # Verify conversion is reversible
    reverse_converted = correction.inverted() @ converted_point
    print(f"Reverse converted: {reverse_converted[:3]}")

def validate_csv_format(csv_path):
    """Validate the CSV file format and warn about issues"""
    try:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            
            # Check for expected header format
            expected_header_start = "matrix_type,m11,m12,m13,m14,m21,m22,m23,m24"
            header_str = ','.join(header)
            if not header_str.startswith(expected_header_start):
                print(f"Warning: CSV header does not match expected format.")
                print(f"Expected: {expected_header_start}...")
                print(f"Found: {header_str}")
            
            # Check for required matrices
            required_matrices = {"modelMatrix", "cameraTransform", "projectionMatrix"}
            found_matrices = set()
            
            for row in reader:
                if len(row) > 0:
                    matrix_type = row[0].strip()
                    if matrix_type in required_matrices:
                        found_matrices.add(matrix_type)
                        
                        # Check if row has 17 elements (type + 16 matrix values)
                        if len(row) != 17:
                            print(f"Warning: {matrix_type} row has {len(row)} values, expected 17")
            
            missing_matrices = required_matrices - found_matrices
            if missing_matrices:
                print(f"Warning: Missing required matrices: {missing_matrices}")
                return False
                
            return True
    except Exception as e:
        print(f"Error validating CSV: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("\n=== Starting ARKit to Blender Conversion with Plane ===")
    
    # Run transformation tests
    test_transforms()
    # Validate CSV format
    if validate_csv_format(csv_path):
        print("CSV validation passed. Setting up scene...")
        setup_scene(
            bg_path=background_path,
            csv_path=csv_path,
            blend_path=output_blend_path,
            render_path=output_image_path,
            plane_size=0.5  # Adjust the plane size as needed
        )
    else:
        print("CSV format validation failed. Please check the CSV file.")