# COMP 557 Assignment 4: Ray Tracing

JunjunXiong
261201887

McGill University

Electrical and Computer Engineering faculty

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
python main.py -i *.json
# change "full_scene = scene_parser.load_scene(f)" 
# into "full_scene = my_scene_parser.load_scene(f)" 
# if you want to generate scene with features added
# this project provide a test json file with added functionalities.
```

# Functionalities
## Basic 
1. Generate Rays
2. Sphere Intersection
3. Lighting and Shading: out/Sphere.png
4. Plane Intersection: out/Plane1.png & out/Plane2.png
5. Shadow Rays: out/TwoSpheresPlane.png
6. Box Intersection: out/BoxRGBLights.png
7. Hierarchy Intersection and Instances: out/BoxStacks.png
8. Triangle Mesh Intersection: TrousMesh.png
9. Anti-Aliasing and Super-Sampling: out/NoAACheckerPlanne.png & out/AACheckerPlane.png
## Sampling and Recursion
10. Reflection and Refraction
```python
    def trace_ray_path_tracing(self, ray, depth, max_depth=5):
        """
        Path tracing method with Monte Carlo integration, including reflection and refraction.
        """
```
11. Motion Blur: Simulate the motion blur effect over motion_blur_samples time steps per pixel. You can adjust this value to increase or decrease the blur effect.For each time sample, the motion_eye_position is updated based on the time_offset, simulating the camera's motion during the exposure period for the pixel. The camera moves by a fixed amount over the frame (e.g., glm.vec3(0, 0, -0.1)), and the motion is sampled at different times (time_offset ranging from 0 to 1).
```python
# Accumulate motion blur over several time steps
                    motion_blur_col = glm.vec3(0, 0, 0)
                    for j in range(motion_blur_samples):
                        # Vary the time offset for motion blur
                        time_offset = j / (motion_blur_samples - 1.0)  # Time steps between 0 and 1
                        motion_eye_position = self.eye_position + time_offset * glm.vec3(0, 0,
                                                                                         -0.1)  # Example camera motion

```
12. Depth of field blur: Apply depth of field blur using the dynamic aperture and focal distance. The ray is computed using the adjusted motion_eye_position for each time step, adding more realism to the final rendered result by blending in the motion blur with the DoF effect.
```python
def get_focus_ray(self, eye_position, ray_dir, aperture, focal_distance)
```
13. Area lights: Increase samples to get smoother shadows.
```python
    def compute_area_light_lighting(self, intersection, light, ray, samples=16):
        """
        Compute lighting contribution for an area light.
        """
```
14. Path tracing:
```python
    def trace_ray_path_tracing(self, ray, depth, max_depth=5):
        """
        Path tracing method with Monte Carlo integration, including reflection and refraction.
        """
```
## Geometry
15. Quadrics
```python
class GeometryQuadrics(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radii: glm.vec3):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radii = radii
```
16. Metaballs
```python
class Metaball(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], position: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.position = position
        self.radius = radius

    def implicit_function(self, p: glm.vec3):
        # Metaball function: f(p) = sum(1 / (|p - center|^2.json))
        return 1.0 / glm.length(p - self.position)
```
17. Bezier surface
```python
class BezierSurface(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], control_points: list[glm.vec3]):
        super().__init__(name, gtype, materials)
        self.control_points = control_points

    def de_casteljau(self, p0, p1, p2, p3, t):
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
```
18. Boolean operations for Constructive Solid Geometry
```python
class CSG(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], left: Geometry, right: Geometry,
                 operation: str):
        super().__init__(name, gtype, materials)
        self.left = left
        self.right = right
        self.operation = operation
```
## Textures
19. Environment maps
```python
class EnvironmentMap(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], texture: str, map_type: str = "cube"):
        super().__init__(name, gtype, materials)
        self.texture = texture  # Path to the texture file (could be a cube map or sphere map)
        self.map_type = map_type  # Type of environment map: "cube" or "sphere"
```
20. Textured mapped surfaces or meshes with adaptive and mipmaps
```python
class TexturedSurface(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], texture: str):
        super().__init__(name, gtype, materials)
        self.texture_path = texture  # Path to the texture file
        self.texture, self.mipmaps = self.load_texture_with_mipmaps(texture)  # Load the texture and mipmaps
```
21. Bump maps with simplex noise
```python
class BumpMappedSurface(TexturedSurface):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], texture: str, bump_strength: float = 0.1):
        super().__init__(name, gtype, materials, texture)
        self.bump_strength = bump_strength  # Strength of bump map (affects normal perturbation)
        self.noise = OpenSimplex()  # Initialize the OpenSimplex noise generator
```
## My Scene
scenes/JunjunXiong-261201887-NovelScene.json
