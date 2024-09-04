import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector

def generate_c2w_matrix(azimuth, elevation, radius):
    # Convert degrees to radians
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    # Calculate camera position in spherical coordinates
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)

    # Camera position (translation)
    translation = np.array([x, y, z])

    # Define the camera orientation (rotation)
    forward = translation / np.linalg.norm(translation)  # Camera looks towards the origin
    right = np.cross(np.array([0, 1, 0]), forward)        # Compute the right vector
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    # Construct the c2w matrix
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = translation

    return c2w

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    return rays_o, rays_d

def generate_rays(image_resolution, intrinsics, c2w):
    h, w = image_resolution
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    directions = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], axis=-1)
    
    rays_o, rays_d = get_rays(directions, c2w)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def ray_cast_mesh(mesh, rays_o, rays_d):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_o,
        ray_directions=rays_d,
        multiple_hits=True,
        return_locations=True
    )
    return index_triangles, index_ray, point_cloud

def get_plane_images(mesh, views, camera_angle_x, max_hits, image_height=512, image_width=512):
    camera_angle_x = float(camera_angle_x)
    
    focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    cx = image_width / 2.0
    cy = image_height / 2.0

    intrinsics = np.array([[focal_length, 0, cx],
                            [0, focal_length, cy],
                            [0, 0, 1]])
    
    all_images = []
    for view_index, view in tqdm(enumerate(views), total=len(views)):
        azimuth, elevation, radius = view['azimuth'], view['elevation'], view['radius']
        c2w = generate_c2w_matrix(azimuth, elevation, radius)
        camera_position = c2w[:3, 3]
        # print(f"Camera Position for View {view_index}: {camera_position}")
        
        rays_o, rays_d = generate_rays((image_height, image_width), intrinsics, c2w)

        index_triangles, index_ray, points = ray_cast_mesh(mesh, rays_o, rays_d)
        normals = mesh.face_normals[index_triangles]
        colors = mesh.visual.face_colors[index_triangles][:, :3] / 255.0

        # Call the visualization function after ray_cast_mesh
        # visualize_rays_and_intersections(mesh, rays_o, rays_d, points, save_dir=save_dir)
        # print(f"Number of rays: {rays_o.shape[0]}")
        # print(f"Number of intersections: {len(points)}")

        # ray to image
        GenDepths = np.ones((max_hits, 1 + 3 + 3 + 1, image_height, image_width), dtype=np.float32)
        GenDepths[:, :4, :, :] = 0 # set depth and normal to zero while color is by default 1 (white)

        hits_per_ray = defaultdict(list)
        for idx, ray_idx in enumerate(index_ray):
            hits_per_ray[ray_idx].append((points[idx], normals[idx], colors[idx], index_triangles[idx]))

        # hits_per_ray[ray_idx].sort(key=lambda hit: np.linalg.norm(hit[0] - c2w[:3, 3]))  # Sort by depth

        # Populate the hit images
        for i in range(max_hits):
            for ray_idx in range(image_height * image_width):
                if i < len(hits_per_ray[ray_idx]):
                    u, v = divmod(ray_idx, image_width)
                    point, normal, color, face_idx = hits_per_ray[ray_idx][i]
                    depth = np.linalg.norm(point - c2w[:3, 3])

                    GenDepths[i, 0, u, v] = depth
                    GenDepths[i, 1:4, u, v] = normal
                    GenDepths[i, 4:7, u, v] = color
                    GenDepths[i, 7, u, v] = face_idx

        images = []
        for i in range(max_hits):
            color_image = (GenDepths[i, 4:7] * 255).astype(np.uint8)
            color_image = np.transpose(color_image, (1, 2, 0)) # (h, w, c)
            images.append(color_image)
        
        all_images.append({
            'view': view,
            'view index': view_index,
            'images': images
        })

    return all_images