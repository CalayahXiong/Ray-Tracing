import random

import glm

class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3):
        self.origin = o
        self.direction = d

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t

class Material:
    def __init__(self, name: str, diffuse: glm.vec3, specular: glm.vec3, shininess: float,
                 reflection: float = 0.0, refraction: float = 0.0, refractive_index: float = 1.0):
        self.name = name
        self.diffuse = diffuse      # kd diffuse coefficient
        self.specular = specular    # ks specular coefficient
        self.shininess = shininess  # specular exponent
        self.reflection = reflection  # Reflection coefficient (0.0 to 1.0)
        self.refraction = refraction  # Refraction coefficient (0.0 to 1.0)
        self.refractive_index = refractive_index  # Refractive index (default 1.0, for air)

class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, attenuation: glm.vec3):
        self.name = name
        self.type = ltype       # type is either "point" or "directional"
        self.colour = colour    # colour and intensity of the light
        self.vector = vector    # position, or normalized direction towards light, depending on the light type
        self.attenuation = attenuation # attenuation coeffs [quadratic, linear, constant] for point lights

class Intersection:
    def __init__(self, t: float, normal: glm.vec3, position: glm.vec3, material: Material):
        self.t = t
        self.normal = normal
        self.position = position
        self.mat = material

    @staticmethod
    def default(): # create an empty intersection record with t = inf
        t = float("inf")
        normal = None 
        position = None 
        mat = None 
        return Intersection(t, normal, position, mat)

class AreaLight:
    def __init__(self, ltype, name, colour, position, size, samples):
        self.type = ltype
        self.name = name
        self.colour = colour
        self.position = position
        self.size = size  # The area size
        self.samples = samples  # Number of samples for light sampling (soft shadows)

    def sample_point(self):
        x_offset = random.uniform(-self.size.x / 2, self.size.x / 2)
        y_offset = random.uniform(-self.size.y / 2, self.size.y / 2)
        return self.position + glm.vec3(x_offset, y_offset, 0)

