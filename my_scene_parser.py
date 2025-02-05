import json

import glm

import geometry as geom
import helperclasses as hc
import my_scene


def make_vec3(array: list):
    return glm.vec3(array[0], array[1], array[2])

def make_matrix( t: glm.vec3, r: glm.vec3, s: glm.vec3 ):
    M = glm.mat4(1.0)
    M = glm.translate( M, t )
    M = glm.rotate( M, glm.radians(r.x), glm.vec3(1, 0, 0) )
    M = glm.rotate( M, glm.radians(r.y), glm.vec3(0, 1, 0) )
    M = glm.rotate( M, glm.radians(r.z), glm.vec3(0, 0, 1) )
    M = glm.scale( M, s )     
    return M   

def load_scene(infile: str):
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)

    # Loading camera
    cam_pos = make_vec3(data["camera"]["position"])
    cam_lookat = make_vec3(data["camera"]["lookAt"])
    cam_up = make_vec3(data["camera"]["up"])
    cam_fov = data["camera"]["fov"]

    # Loading resolution
    default_resolution = [1280, 720]
    width = data.get("resolution", default_resolution)[0]
    height = data.get("resolution", default_resolution)[1]

    # Loading ambient light
    ambient = make_vec3(data.get("ambient", [0.1, 0.1, 0.2]))

    # Loading Anti-Aliasing options
    jitter = data.get("AA_jitter", False)
    samples = data.get("AA_samples", 1)

    # Loading scene lights
    lights = []
    for light in data.get("lights", []):
        l_type = light["type"]
        l_name = light["name"]
        l_colour = make_vec3(light["colour"])
        l_power = light.get("power", 1.0)

        if l_type == "point":
            l_vector = make_vec3(light["position"])
            l_attenuation = glm.vec3(1, 0, 0) if "attenuation" not in light else make_vec3(light["attenuation"])
            lights.append(hc.Light(l_type, l_name, l_colour * l_power, l_vector, l_attenuation))

        elif l_type == "directional":
            l_vector = glm.normalize(make_vec3(light["direction"]))
            l_attenuation = glm.vec3(0, 0, 0)
            if "attenuation" in light:
                print("Directional light", l_name, "has attenuation, ignoring")
            lights.append(hc.Light(l_type, l_name, l_colour * l_power, l_vector, l_attenuation))

        elif l_type == "area":
            # Handling area lights
            l_position = make_vec3(light["position"])

            # Ensure size has 3 elements, else fallback to default value
            l_size = glm.vec3(1, 1, 0)  # Default fallback value
            if "size" in light:
                try:
                    l_size = make_vec3(light["size"])
                except IndexError:
                    print(f"Warning: Invalid size value for area light '{l_name}', using default size.")

            l_samples = light.get("samples", 4)  # Default to 4 samples if not specified
            lights.append(hc.AreaLight(l_type, l_name, l_colour * l_power, l_position, l_size, l_samples))

        else:
            print(f"Unknown light type '{l_type}', skipping initialization.")
            continue

    # Loading materials
    material_by_name = {}
    for material in data["materials"]:
        mat_name = material["name"]
        mat_diffuse = make_vec3(material["diffuse"])
        mat_specular = make_vec3(material["specular"])
        mat_shininess = material.get("shininess", 0)
        mat_reflection = material.get("reflection", 0.0)
        mat_refraction = material.get("refraction", 0.0)
        mat_refr_index = material.get("refraction_index", 1.0)
        material_by_name[mat_name] = hc.Material(mat_name, mat_diffuse, mat_specular, mat_shininess, mat_reflection, mat_refraction, mat_refr_index)

    # Load geometries
    objects = []
    geometry_by_name = {}

    for geometry in data["objects"]:
        g = load_geometry(geometry, material_by_name, geometry_by_name)
        if g is not None:
            objects.append(g)
            geometry_by_name[g.name] = g

    # Read aperture and focal_distance from JSON
    aperture = data["effects"].get("depth_of_field", {}).get("aperture", 0.1)
    focal_distance = data["effects"].get("depth_of_field", {}).get("focal_distance", 5.0)

    # Create the scene and pass aperture and focal_distance to it
    return my_scene.Scene(
        width, height, jitter, samples,  # General settings
        cam_pos, cam_lookat, cam_up, cam_fov,  # Camera settings
        ambient, lights,  # Light settings
        objects, aperture, focal_distance  # Geometry and DOF settings
    )

def load_geometry( geometry, material_by_name, geometry_by_name ):

    # Elements common to all objects: name, type, and material(s)
    g_name = geometry["name"]
    g_type = geometry["type"]
    g_mats = [ material_by_name[mat] for mat in geometry.get("materials",[]) ]

    if g_type == "sphere":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_radius = geometry["radius"]
        return geom.Sphere(g_name, g_type, g_mats, g_pos, g_radius)
    elif g_type == "plane":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_normal = make_vec3(geometry["normal"])
        return geom.Plane(g_name, g_type, g_mats, g_pos, g_normal)
    elif g_type == "box":
        minpos = make_vec3(geometry.get("min",[-1,-1,-1]))
        maxpos = make_vec3(geometry.get("max",[1,1,1]))
        return geom.AABB(g_name, g_type, g_mats, minpos, maxpos)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_scale = geometry["scale"]
        return geom.Mesh(g_name, g_type, g_mats, g_pos, g_scale, g_path)

    elif g_type == "ellipsoid":
        g_center = make_vec3(geometry.get("center", [0, 0, 0]))
        g_radii = make_vec3(geometry.get("radii", [1, 1, 1]))
        return geom.GeometryQuadrics(g_name, g_type, g_mats, g_center, g_radii)

    elif g_type == "instance":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_r = make_vec3(geometry.get("rotation", [0, 0, 0]))
        g_s = make_vec3(geometry.get("scale", [1, 1, 1]))
        M = make_matrix(g_pos, g_r, g_s)
        node = geom.Node(g_name, g_type, M, g_mats)
        node.children.append( geometry_by_name[geometry["ref"]] )
        return node

    elif g_type == "node":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_r = make_vec3(geometry.get("rotation", [0, 0, 0]))
        g_s = make_vec3(geometry.get("scale", [1, 1, 1]))
        M = make_matrix(g_pos, g_r, g_s)
        node = geom.Node(g_name, g_type, M, g_mats)
        for child in geometry["children"]:
            g = load_geometry(child, material_by_name, geometry_by_name)
            node.children.append(g)
            geometry_by_name[g.name] = g
        return node
    elif g_type == "cone":
        g_apex = make_vec3(geometry.get("apex", [0, 1, 0]))
        g_base_center = make_vec3(geometry.get("base_center", [0, 0, 0]))
        g_radius = geometry["radius"]
        return geom.Cone(g_name, g_type, g_mats, g_apex, g_base_center, g_radius)
    elif g_type == "cylinder":
        g_base_center = make_vec3(geometry.get("base_center", [0, 0, 0]))
        g_top_center = make_vec3(geometry.get("top_center", [0, 1, 0]))
        g_radius = geometry["radius"]
        return geom.Cylinder(g_name, g_type, g_mats, g_base_center, g_top_center, g_radius)

    elif g_type == "csg":
        # Parse the CSG operation type (union, intersection, or difference)
        operation = geometry["operation"].lower()  # Ensure operation is lowercase for consistency

        #left_name = geometry["left"]
        #right_name = geometry["right"]

        # Load the left and right operands for the CSG operation
        left = load_geometry(geometry["left"], material_by_name, geometry_by_name)
        right = load_geometry(geometry["right"], material_by_name, geometry_by_name)

        # Ensure that both left and right operands are valid geometry objects
        if left is None or right is None:
            print(f"Error: CSG operands '{geometry['left']}' or '{geometry['right']}' not found!")
            return None

        # Return the CSG object
        return geom.CSG(g_name, g_type, g_mats, left, right, operation)

    else:
        print("Unkown object type", g_type, ", skipping initialization")
        return None