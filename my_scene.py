import math
import random

import glm
import numpy as np
from tqdm import tqdm

import helperclasses as hc


class Scene:

    def __init__(self, width, height, jitter, samples, eye_position, lookat, up, fov, ambient, lights, objects, aperture, focal_distance):
        self.width = width
        self.height = height
        self.aspect = width / height
        self.jitter = jitter
        self.samples = samples
        self.eye_position = eye_position
        self.lookat = lookat
        self.up = up
        self.fov = fov
        self.ambient = ambient
        self.lights = lights
        self.objects = objects
        self.aperture = aperture  # Dynamic aperture value
        self.focal_distance = focal_distance  # Dynamic focal distance
        self.max_recursion_depth = 5  # Prevent infinite recursion for reflective rays

    def render(self):
        image = np.zeros((self.height, self.width, 3))
        cam_dir = self.eye_position - self.lookat
        distance_to_plane = 1.0
        top = distance_to_plane * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        motion_blur_samples = 4  # Number of motion blur time samples per pixel
        for col in tqdm(range(self.width)):
            for row in range(self.height):
                colour = glm.vec3(0, 0, 0)

                for i in range(self.samples):
                    x_offset, y_offset = self.get_sample_offset(i % self.samples, i // self.samples, self.samples)
                    u_screen = left + (right - left) * (col + x_offset) / self.width
                    v_screen = bottom + (top - bottom) * (row + y_offset) / self.height
                    ray_dir = glm.normalize(u_screen * u + v_screen * v - distance_to_plane * w)

                    # Accumulate motion blur over several time steps
                    motion_blur_col = glm.vec3(0, 0, 0)
                    for j in range(motion_blur_samples):
                        # Vary the time offset for motion blur
                        time_offset = j / (motion_blur_samples - 1.0)  # Time steps between 0 and 1
                        motion_eye_position = self.eye_position + time_offset * glm.vec3(0, 0,
                                                                                         -0.1)  # Example camera motion

                        # Apply depth of field blur using the dynamic aperture and focal distance
                        ray = self.get_focus_ray(motion_eye_position, ray_dir)

                        # Trace the ray and accumulate the results
                        motion_blur_col += self.trace_ray_path_tracing(ray, depth=0)

                    # Average the motion blur over time samples
                    colour += motion_blur_col / motion_blur_samples

                image[row, col, 0] = max(0.0, min(1.0, colour.x / self.samples))
                image[row, col, 1] = max(0.0, min(1.0, colour.y / self.samples))
                image[row, col, 2] = max(0.0, min(1.0, colour.z / self.samples))

        image = np.flipud(image)
        return image

    def trace_ray_path_tracing(self, ray, depth, max_depth=5):
        """
        Path tracing method with Monte Carlo integration, including reflection and refraction.
        """
        if depth > max_depth:
            return glm.vec3(0, 0, 0)  # Terminate recursion

        closest_intersection = hc.Intersection.default()
        for obj in self.objects:
            obj.intersect(ray, closest_intersection)

        if closest_intersection.t == float('inf'):
            return glm.vec3(0, 0, 0)  # Background color

        # Compute direct lighting (area lights included)
        direct_lighting = glm.vec3(0, 0, 0)
        for light in self.lights:
            if light.type == "area":
                direct_lighting += self.compute_area_light_lighting(closest_intersection, light, ray, samples=16)
            else:
                direct_lighting += self.compute_lighting(closest_intersection, light, ray)

        # Indirect lighting via Monte Carlo sampling
        mat = closest_intersection.mat
        if depth < max_depth and glm.length(mat.diffuse + mat.specular) > 0:
            # Sample a new direction using cosine-weighted hemisphere sampling
            normal = closest_intersection.normal
            sample_dir = self.sample_hemisphere(normal)
            sample_ray = hc.Ray(o=closest_intersection.position + 1e-4 * normal, d=sample_dir)
            indirect_lighting = mat.diffuse * self.trace_ray_path_tracing(sample_ray, depth + 1, max_depth)
        else:
            indirect_lighting = glm.vec3(0, 0, 0)

        # Reflection and Refraction (if applicable)
        if mat.reflection > 0:
            reflection_ray = self.reflect_ray(ray, closest_intersection)
            reflection = mat.reflection * self.trace_ray_path_tracing(reflection_ray, depth + 1, max_depth)
        else:
            reflection = glm.vec3(0, 0, 0)

        if mat.refraction > 0:
            refraction_ray = self.refract_ray(ray, closest_intersection, mat.refractive_index)
            refraction = mat.refraction * self.trace_ray_path_tracing(refraction_ray, depth + 1, max_depth)
        else:
            refraction = glm.vec3(0, 0, 0)

        return direct_lighting + indirect_lighting + reflection + refraction

    def reflect_ray(self, ray, intersection):
        """
        Reflect the ray at the intersection point.
        """
        normal = intersection.normal
        ray_dir = glm.normalize(ray.direction)
        reflect_dir = ray_dir - 2 * glm.dot(ray_dir, normal) * normal
        reflect_origin = intersection.position + 1e-4 * normal  # Small offset to prevent self-intersection
        return hc.Ray(o=reflect_origin, d=reflect_dir)

    def refract_ray(self, ray, intersection, refraction_index):
        """
        Refract the ray at the intersection point based on the material's refraction index.
        """
        normal = intersection.normal
        ray_dir = glm.normalize(ray.direction)
        cos_theta = glm.dot(normal, -ray_dir)
        refract_dir = glm.vec3(0, 0, 0)

        if cos_theta > 0:
            n1 = 1.0  # Air refraction index
            n2 = refraction_index
        else:
            n1 = refraction_index
            n2 = 1.0

        eta = n1 / n2
        k = 1 - eta * eta * (1 - cos_theta * cos_theta)
        if k > 0:
            refract_dir = eta * ray_dir + (eta * cos_theta - math.sqrt(k)) * normal
        refract_origin = intersection.position + 1e-4 * normal  # Small offset to prevent self-intersection
        return hc.Ray(o=refract_origin, d=refract_dir)

    def compute_area_light_lighting(self, intersection, light, ray, samples=16):
        """
        Compute lighting contribution for an area light.
        """
        if not isinstance(light, hc.AreaLight):
            print(f"Warning: Light '{light.name}' is not an AreaLight. Skipping.")
            return glm.vec3(0, 0, 0)

        normal = intersection.normal
        position = intersection.position
        mat = intersection.mat

        diffuse = glm.vec3(0, 0, 0)
        specular = glm.vec3(0, 0, 0)

        for _ in range(samples):
            # Sample a random point on the area light's surface
            light_point = light.sample_point()
            light_dir = glm.normalize(light_point - position)
            distance_to_light = glm.length(light_point - position)

            shadow_ray_origin = position + 1e-4 * normal
            shadow_ray = hc.Ray(o=shadow_ray_origin, d=light_dir)

            shadow_intersection = hc.Intersection.default()

            shadow_occluded = any(
                obj.intersect(shadow_ray, shadow_intersection) and shadow_intersection.t < distance_to_light
                for obj in self.objects
            )
            if shadow_occluded:
                continue

            ndotl = glm.dot(normal, light_dir)
            if ndotl > 0:
                diffuse += mat.diffuse * light.colour * ndotl / samples

                if glm.length(mat.specular) > 0:
                    view_dir = glm.normalize(ray.origin - position)
                    half_vector = glm.normalize(light_dir + view_dir)
                    ndoth = glm.dot(normal, half_vector)
                    if ndoth > 0:
                        specular += mat.specular * light.colour * (ndoth ** mat.shininess) / samples

        return diffuse + specular

    def compute_lighting(self, intersection, light, ray):
        if intersection is None:
            return glm.vec3(0, 0, 0)  # No intersection, no lighting contribution

        normal = intersection.normal
        position = intersection.position
        mat = intersection.mat

        diffuse = glm.vec3(0, 0, 0)
        specular = glm.vec3(0, 0, 0)

        light_dir = glm.normalize(light.vector - position)
        distance_to_light = glm.length(light.vector - position)
        shadow_ray_origin = position + 1e-4 * normal  # To prevent self-intersection
        shadow_ray = hc.Ray(o=shadow_ray_origin, d=light_dir)

        shadow_intersection = hc.Intersection.default()
        shadow_occluded = any(
            obj.intersect(shadow_ray, shadow_intersection) and shadow_intersection.t < distance_to_light
            for obj in self.objects
        )
        if shadow_occluded:
            return glm.vec3(0, 0, 0)  # Light is blocked, no contribution

        ndotl = glm.dot(normal, light_dir)
        if ndotl > 0:
            diffuse += mat.diffuse * light.colour * ndotl

            if glm.length(mat.specular) > 0:
                view_dir = glm.normalize(ray.origin - position)
                half_vector = glm.normalize(light_dir + view_dir)
                ndoth = glm.dot(normal, half_vector)
                if ndoth > 0:
                    specular += mat.specular * light.colour * (ndoth ** mat.shininess)

        return diffuse + specular

    def get_sample_offset(self, i, j, total_samples):
        """
        Random jitter offset for sample positions to reduce aliasing.
        """
        return random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)

    def get_focus_ray(self, eye_position, ray_dir):
        # Calculate depth of field based on the dynamic aperture and focal distance
        lens_radius = self.aperture * 0.5  # Size of the aperture
        theta = random.uniform(0, 2 * math.pi)
        r = lens_radius * math.sqrt(random.uniform(0, 1))
        offset = glm.vec3(r * math.cos(theta), r * math.sin(theta), 0)

        # Focus point based on focal distance
        focus_point = eye_position + self.focal_distance * ray_dir
        ray_origin = eye_position + offset
        ray_direction = glm.normalize(focus_point - ray_origin)
        return hc.Ray(o=ray_origin, d=ray_direction)

    def sample_hemisphere(self, normal):
        """
        Sample a direction from a cosine-weighted hemisphere.
        """
        z = random.uniform(0, 1)
        theta = random.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - z * z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        sampled_dir = glm.vec3(x, y, z)

        if glm.dot(sampled_dir, normal) < 0:
            sampled_dir = -sampled_dir

        return glm.normalize(sampled_dir)
