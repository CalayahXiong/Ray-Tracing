import math

from opensimplex import OpenSimplex

import helperclasses as hc
import glm
import igl
import numpy as np
from PIL import Image

class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):

        # TODO: Create intersect code for Sphere
        # Ray origin and direction
        o = ray.origin
        d = ray.direction

        # Vector from ray origin to sphere center
        oc = o - self.center

        # Coefficients of the quadratic equation
        a = glm.dot(d, d)
        b = 2.0 * glm.dot(d, oc)
        c = glm.dot(oc, oc) - self.radius ** 2

        if a == 0:
            return None

        # Discriminant
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None # No intersection

        # Calculate the two roots of the quadratic
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # Select the smallest positive t
        if t1 > 0 and (t1 < intersect.t or intersect.t == float("inf")):
            intersect.t = t1
            intersect.position = o + t1 * d
            intersect.normal = glm.normalize(intersect.position - self.center)
            intersect.mat = self.materials[0]
            return intersect

        if t2 > 0 and t2 < intersect.t:
            intersect.t = t2
            intersect.position = o + t2 * d
            intersect.normal = glm.normalize(intersect.position - self.center)
            intersect.mat = self.materials[0]
            return intersect

        # If no intersection found, return False
        return None


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):

        # TODO: Create intersect code for Plane
        # Compute the denominator (dot product of ray direction and plane normal)
        denom = glm.dot(ray.direction, self.normal)

        # If the denominator is close to zero, the ray is parallel to the plane
        if abs(denom) < 1e-6:  # A small threshold to check if ray is parallel
            return False

        # Compute the vector from the ray origin to a point on the plane
        diff = self.point - ray.origin

        # Compute the t value using the formula
        t = glm.dot(diff, self.normal) / denom

        # If t is negative, the intersection is behind the ray origin
        if t < 0:
            return False

        # Update the intersection if this is the closest so far
        if t < intersect.t:
            intersect.t = t
            intersect.position = ray.origin + t * ray.direction
            intersect.normal = glm.normalize(self.normal)

            # Compute checkerboard pattern
            x = intersect.position.x
            z = intersect.position.z
            checkerboard = int(glm.floor(x) + glm.floor(z)) % 2

            # Assign the material based on the checkerboard pattern
            if len(self.materials) > 1 and checkerboard != 0:
                intersect.mat = self.materials[1]  # Use second material
            else:
                intersect.mat = self.materials[0]  # Use first material

            return intersect

        return None  # No intersection


class Cone(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], apex: glm.vec3, base_center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.apex = apex
        self.base_center = base_center
        self.radius = radius
        self.axis = glm.normalize(base_center - apex)
        self.height = glm.length(base_center - apex)
        self.tan_half_angle = (radius / self.height) ** 2  # Square of tangent of half-angle for cone

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # Vector from ray origin to cone apex
        v_apex = ray.origin - self.apex

        # Dot products for quadratic equation
        d_dot_axis = glm.dot(ray.direction, self.axis)
        v_dot_axis = glm.dot(v_apex, self.axis)

        # Components of the quadratic equation
        a = glm.dot(ray.direction, ray.direction) - (1 + self.tan_half_angle) * d_dot_axis ** 2
        b = 2 * (glm.dot(ray.direction, v_apex) - (1 + self.tan_half_angle) * d_dot_axis * v_dot_axis)
        c = glm.dot(v_apex, v_apex) - (1 + self.tan_half_angle) * v_dot_axis ** 2

        if abs(a) < 1e-6:
            return None

        # Solve the quadratic equation
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False

        sqrt_discriminant = glm.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        # Select the closest positive t value
        t = t1 if t1 > 0 else t2
        if t < 0 or t >= intersect.t:
            return False

        # Compute intersection position
        position = ray.origin + t * ray.direction

        # Ensure the intersection is within the cone's height
        height_check = glm.dot(position - self.apex, self.axis)
        if height_check < 0 or height_check > self.height:
            return False

        # Update the intersection record
        intersect.t = t
        intersect.position = position
        intersect.normal = glm.normalize(position - self.apex - self.axis * glm.dot(position - self.apex, self.axis))
        intersect.mat = self.materials[0]
        return True

class Cylinder(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], base_center: glm.vec3, top_center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.base_center = base_center
        self.top_center = top_center
        self.radius = radius
        self.axis = glm.normalize(top_center - base_center)
        self.height = glm.length(top_center - base_center)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # Vector from base center to ray origin
        v_base = ray.origin - self.base_center

        # Projection of ray direction and base vector onto the cylinder axis
        d_dot_axis = glm.dot(ray.direction, self.axis)
        v_dot_axis = glm.dot(v_base, self.axis)

        # Perpendicular components for quadratic equation
        d_perp = ray.direction - d_dot_axis * self.axis
        v_perp = v_base - v_dot_axis * self.axis

        # Quadratic equation components
        a = glm.dot(d_perp, d_perp)
        b = 2 * glm.dot(d_perp, v_perp)
        c = glm.dot(v_perp, v_perp) - self.radius ** 2

        if abs(a) < 1e-6:
            return None

        # Solve the quadratic equation
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False

        sqrt_discriminant = glm.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        # Select the closest positive t value
        t = t1 if t1 > 0 else t2
        if t < 0 or t >= intersect.t:
            return False

        # Compute intersection position
        position = ray.origin + t * ray.direction

        # Check if the intersection is within the cylinder's height
        height_check = glm.dot(position - self.base_center, self.axis)
        if height_check < 0 or height_check > self.height:
            return False

        # Update the intersection record
        intersect.t = t
        intersect.position = position
        intersect.normal = glm.normalize(v_perp + t * d_perp)
        intersect.mat = self.materials[0]
        return True

class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], minpos: glm.vec3, maxpos: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        self.minpos = minpos
        self.maxpos = maxpos

    def calculate_normal(self, position):
        # Calculate the normal based on the closest face
        epsilon = 1e-4  # Small tolerance for numerical errors
        if abs(position.x - self.minpos.x) < epsilon:
            return glm.vec3(-1, 0, 0)
        elif abs(position.x - self.maxpos.x) < epsilon:
            return glm.vec3(1, 0, 0)
        elif abs(position.y - self.minpos.y) < epsilon:
            return glm.vec3(0, -1, 0)
        elif abs(position.y - self.maxpos.y) < epsilon:
            return glm.vec3(0, 1, 0)
        elif abs(position.z - self.minpos.z) < epsilon:
            return glm.vec3(0, 0, -1)
        elif abs(position.z - self.maxpos.z) < epsilon:
            return glm.vec3(0, 0, 1)
        return glm.vec3(0, 0, 0)  # Fallback (should not occur)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):

        # TODO: Create intersect code for Cube
        # Initialize t_min and t_max to represent the ray's range along each axis
        t_min = (self.minpos.x - ray.origin.x) / ray.direction.x if ray.direction.x != 0 else -float('inf')
        t_max = (self.maxpos.x - ray.origin.x) / ray.direction.x if ray.direction.x != 0 else float('inf')
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        ty_min = (self.minpos.y - ray.origin.y) / ray.direction.y if ray.direction.y != 0 else -float('inf')
        ty_max = (self.maxpos.y - ray.origin.y) / ray.direction.y if ray.direction.y != 0 else float('inf')
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min

        # Update t_min and t_max with the y-axis intersections
        if t_min > ty_max or ty_min > t_max:
            return None  # No intersection
        t_min = max(t_min, ty_min)
        t_max = min(t_max, ty_max)

        tz_min = (self.minpos.z - ray.origin.z) / ray.direction.z if ray.direction.z != 0 else -float('inf')
        tz_max = (self.maxpos.z - ray.origin.z) / ray.direction.z if ray.direction.z != 0 else float('inf')
        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min

        # Update t_min and t_max with the z-axis intersections
        if t_min > tz_max or tz_min > t_max:
            return False  # No intersection
        t_min = max(t_min, tz_min)
        t_max = min(t_max, tz_max)

        # If t_min is negative, the box is behind the ray origin
        if t_max < 0:
            return None

        # Record the intersection if it is closer than the current closest intersection
        if t_min < intersect.t:
            intersect.t = t_min
            intersect.position = ray.origin + t_min * ray.direction
            intersect.normal = self.calculate_normal(intersect.position)
            intersect.mat = self.materials[0]  # Assuming a single material for the box
            return intersect

        return None


class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):

        # TODO: Create intersect code for Mesh
        closest_t = float('inf')  # Track the closest intersection
        hit = False  # Flag to indicate whether a triangle was hit

        # Loop through each face (triangle) in the mesh
        for face in self.faces:
            # Retrieve vertices of the triangle
            v0 = self.verts[face[0]]
            v1 = self.verts[face[1]]
            v2 = self.verts[face[2]]

            # Edges of the triangle
            edge1 = v1 - v0
            edge2 = v2 - v0

            # Calculate determinant (a) to test for ray-triangle intersection
            h = glm.cross(ray.direction, edge2)
            a = glm.dot(edge1, h)

            if abs(a) < 1e-6:
                continue  # Ray is parallel to the triangle

            f = 1.0 / a
            s = ray.origin - v0
            u = f * glm.dot(s, h)

            if u < 0.0 or u > 1.0:
                continue  # Intersection point is outside the triangle

            q = glm.cross(s, edge1)
            v = f * glm.dot(ray.direction, q)

            if v < 0.0 or u + v > 1.0:
                continue  # Intersection point is outside the triangle

            # Compute the intersection point (t)
            t = f * glm.dot(edge2, q)

            if t > 1e-6 and t < closest_t:  # Valid intersection and closer than previous
                closest_t = t
                #hit = True

                # Update the intersection record
                intersect.t = t
                intersect.position = ray.origin + t * ray.direction
                intersect.normal = glm.normalize(glm.cross(edge1, edge2))  # Flat shading: normal of the triangle
                intersect.mat = self.materials[0]  # Assign material

        if closest_t != float('inf'):
            return intersect
        else:
            return None


class Node(Geometry):
    def __init__(self, name: str, gtype: str, M: glm.mat4, materials: list[hc.Material]):
        super().__init__(name, gtype, materials)
        self.children: list[Geometry] = []
        self.M = M
        self.Minv = glm.inverse(M)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # only the node in the middle (and only the node near the floor) show shadow
        # Why node upon and near the middle without shadow?
        # Shadow checking is not iterate to them?

        # TODO: Create intersect code for Node
        """
        Perform intersection by transforming the ray into local space,
        intersecting child nodes, and transforming the result back to world space.

        Returns:
            bool: True if an intersection occurred, False otherwise.
        """
        # Transform ray to local space
        local_origin = self.Minv * glm.vec4(ray.origin, 1.0)
        local_direction = self.Minv * glm.vec4(ray.direction, 0.0)
        local_ray = hc.Ray(glm.vec3(local_origin), glm.vec3(local_direction))

        hit = False  # Track if any intersection occurs

        # Check intersections for this node and its children
        for child in self.children:
            child_intersection = hc.Intersection.default()
            if child.intersect(local_ray, child_intersection):
                hit = True
                # If the child's intersection is closer, update the main intersection
                if child_intersection.t < intersect.t:
                    intersect.t = child_intersection.t
                    intersect.position = glm.vec3(self.M * glm.vec4(child_intersection.position, 1.0))
                    intersect.normal = glm.normalize(glm.vec3(glm.transpose(self.Minv) * glm.vec4(child_intersection.normal, 0.0)))
                    intersect.mat = child_intersection.mat or (self.materials[0] if self.materials else None)

        # Assign the material of this node if no child provides one
        if hit and intersect.mat is None and self.materials:
            intersect.mat = self.materials[0]

        return hit


class GeometryQuadrics(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radii: glm.vec3):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radii = radii  # Radii for ellipsoid (x, y, z)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        o = ray.origin
        d = ray.direction
        oc = o - self.center

        # Ellipsoid equation: (x/a)^2.json + (y/b)^2.json + (z/c)^2.json = 1
        a, b, c = self.radii.x, self.radii.y, self.radii.z
        a2, b2, c2 = a * a, b * b, c * c

        # Coefficients for the quadratic equation
        A = (d.x * d.x) / a2 + (d.y * d.y) / b2 + (d.z * d.z) / c2
        B = 2 * ((oc.x * d.x) / a2 + (oc.y * d.y) / b2 + (oc.z * d.z) / c2)
        C = (oc.x * oc.x) / a2 + (oc.y * oc.y) / b2 + (oc.z * oc.z) / c2 - 1

        if abs(A) < 1e-6:
            return None

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            return None  # No intersection

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        # Select the smallest positive t
        t = min(t1, t2)
        if t > 0 and t < intersect.t:
            intersect.t = t
            intersect.position = o + t * d
            # Compute normal for the ellipsoid
            intersect.normal = glm.normalize(glm.vec3((intersect.position.x - self.center.x) / a2,
                                                      (intersect.position.y - self.center.y) / b2,
                                                      (intersect.position.z - self.center.z) / c2))
            intersect.mat = self.materials[0]
            return intersect
        return None


class Metaball(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], position: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.position = position
        self.radius = radius

    def implicit_function(self, p: glm.vec3):
        # Metaball function: f(p) = sum(1 / (|p - center|^2.json))
        return 1.0 / glm.length(p - self.position)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # Approximate intersection by sampling the field along the ray
        # For simplicity, we will calculate intersections based on a threshold of implicit function value
        step_size = 0.1
        for t in range(1000):  # Try 1000 steps along the ray
            point = ray.origin + t * step_size * ray.direction
            if self.implicit_function(point) < 0.1:
                intersect.t = t * step_size
                intersect.position = point
                intersect.normal = glm.normalize(point - self.position)
                intersect.mat = self.materials[0]
                return intersect
        return None


class BezierSurface(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], control_points: list[glm.vec3]):
        super().__init__(name, gtype, materials)
        self.control_points = control_points

    def de_casteljau(self, p0, p1, p2, p3, t):
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # Example: Intersect with a simplified bezier patch
        # We'll compute the intersection based on ray parameterization and control points
        # For simplicity, we'll just return the control point for now
        return intersect  # Replace with actual ray-surface intersection logic


class CSG(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], left: Geometry, right: Geometry,
                 operation: str):
        super().__init__(name, gtype, materials)
        self.left = left
        self.right = right
        self.operation = operation

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        left_intersect = self.left.intersect(ray, hc.Intersection.default())
        right_intersect = self.right.intersect(ray, hc.Intersection.default())

        if self.operation == "union":
            if left_intersect and (not right_intersect or left_intersect.t < right_intersect.t):
                return left_intersect
            elif right_intersect:
                return right_intersect
        elif self.operation == "intersection":
            if left_intersect and right_intersect:
                return left_intersect if left_intersect.t < right_intersect.t else right_intersect
        elif self.operation == "difference":
            if left_intersect and not right_intersect:
                return left_intersect
        return None


class EnvironmentMap(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], texture: str, map_type: str = "cube"):
        super().__init__(name, gtype, materials)
        self.texture = texture  # Path to the texture file (could be a cube map or sphere map)
        self.map_type = map_type  # Type of environment map: "cube" or "sphere"

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # For simplicity, assume a sphere map (cube map would involve more complex intersection logic)
        if self.map_type == "sphere":
            # Compute the intersection with the sphere (assuming a unit sphere for environment mapping)
            o = ray.origin
            d = glm.normalize(ray.direction)
            radius = 1.0  # Assuming unit sphere for environment map

            # Intersection with a sphere
            t = glm.dot(o, d) + glm.sqrt(glm.dot(o, o) - glm.dot(d, d) * glm.dot(o, o))

            if t > 0:
                intersect.t = t
                intersect.position = o + t * d
                # Map texture coordinates based on intersection point on the sphere
                normal = glm.normalize(intersect.position)
                # Placeholder texture lookup (this can be replaced with actual texture mapping logic)
                intersect.mat = self.materials[0]
                return intersect
        return None


class TexturedSurface(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], texture: str):
        super().__init__(name, gtype, materials)
        self.texture_path = texture  # Path to the texture file
        self.texture, self.mipmaps = self.load_texture_with_mipmaps(texture)  # Load the texture and mipmaps

    def load_texture_with_mipmaps(self, texture_path: str):
        """Loads the texture image and generates mipmaps."""
        texture = Image.open(texture_path)
        mipmaps = [texture]
        # Generate mipmaps
        while mipmaps[-1].size[0] > 1 and mipmaps[-1].size[1] > 1:
            mipmaps.append(mipmaps[-1].resize((mipmaps[-1].size[0] // 2, mipmaps[-1].size[1] // 2), Image.ANTIALIAS))
        return texture, mipmaps

    def get_texture(self, uv: glm.vec2, lod: int = 0):
        """Samples the texture at given UV coordinates with mipmaps and adaptive sampling."""
        width, height = self.mipmaps[lod].size
        u_pixel = int(uv.x * width)  # Map UV to pixel coordinates
        v_pixel = int((1 - uv.y) * height)  # Invert the v-coordinate (Y is flipped in image space)
        color = np.array(self.mipmaps[lod].getpixel((u_pixel, v_pixel))) / 255.0
        return glm.vec3(color[0], color[1], color[2])

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        """Perform intersection test and compute UV coordinates."""
        if super().intersect(ray, intersect):

            uv = self.calculate_uv_coordinates(intersect.position)

            screen_space_size = self.calculate_screen_space_size(intersect.position, ray)
            lod = self.calculate_lod(screen_space_size)

            intersect.mat = self.get_texture(uv, lod)
            return intersect
        return None

    def calculate_uv_coordinates(self, position: glm.vec3):
        """Placeholder for UV coordinate calculation. For simplicity, we'll assume spherical UV mapping."""
        theta = np.arctan2(position.z, position.x)
        phi = np.arccos(position.y / glm.length(position))

        u = (theta + np.pi) / (2 * np.pi)  # Map theta to [0, 1]
        v = (phi) / np.pi  # Map phi to [0, 1]
        return glm.vec2(u, v)

    def calculate_screen_space_size(self, position: glm.vec3, ray: hc.Ray):
        """Estimate the screen-space size of the intersection (this will be used for adaptive sampling)."""
        screen_space_size = glm.length(position - ray.o) * glm.length(ray.d)
        return screen_space_size

    def calculate_lod(self, screen_space_size: float):
        """Calculate the level of detail (LOD) based on screen-space size."""
        lod = int(math.log(screen_space_size + 1) / math.log(2))  # log2-based LOD selection
        lod = min(len(self.mipmaps) - 1, max(0, lod))  # Clamp LOD to valid range
        return lod


class BumpMappedSurface(TexturedSurface):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], texture: str, bump_strength: float = 0.1):
        super().__init__(name, gtype, materials, texture)
        self.bump_strength = bump_strength  # Strength of bump map (affects normal perturbation)
        self.noise = OpenSimplex()  # Initialize the OpenSimplex noise generator

    def generate_bump_map(self, uv: glm.vec2):
        # Generate a bump map using OpenSimplex noise
        # OpenSimplex noise has better spatial coherence than Perlin noise
        noise_value = self.noise.noise2d(uv.x * 10.0, uv.y * 10.0)  # Scale UV to get different noise detail
        return glm.vec3(0.0, 0.0, noise_value * self.bump_strength)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        if super().intersect(ray, intersect):
            # Perturb the normal based on the bump map at the intersection point
            uv = glm.vec2(0.5, 0.5)  # Example UV coordinates; in real cases, this would come from texture mapping
            bump_normal = self.generate_bump_map(uv)
            intersect.normal += bump_normal  # Modify normal based on bump map
            intersect.normal = glm.normalize(intersect.normal)  # Normalize the normal
            return intersect
        return None



