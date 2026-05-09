"""Mesh post-processing utilities (pymeshlab wrappers)."""

import numpy as np
import pymeshlab as pml


def decimate_mesh(verts, faces, target,
                  backend='pymeshlab',
                  remesh=False,
                  optimal_placement=True):
    """Reduce ``faces`` to approximately ``target`` triangles.

    Args:
        backend: ``'pymeshlab'`` or ``'pyfqmr'``.
        optimal_placement: pass ``False`` for flat meshes to avoid spike artifacts.
    """
    ori_vert_shape = verts.shape
    ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, _ = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh')

        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimal_placement,
        )
        if remesh:
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f'[INFO] mesh decimation: {ori_vert_shape} --> {verts.shape}, '
          f'{ori_face_shape} --> {faces.shape}')
    return verts, faces


def clean_mesh(verts, faces,
               v_pct: float = 1,
               min_f: int = 8,
               min_d: float = 5,
               repair: bool = True,
               remesh: bool = True,
               remesh_size: float = 0.01):
    """Remove unreferenced/duplicate elements and optionally remesh."""
    ori_vert_shape = verts.shape
    ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')

    ms.meshing_remove_unreferenced_vertices()
    if v_pct > 0:
        ms.meshing_merge_close_vertices(threshold=pml.Percentage(v_pct))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.Percentage(min_d),
        )
    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.AbsoluteValue(remesh_size),
        )

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {ori_vert_shape} --> {verts.shape}, '
          f'{ori_face_shape} --> {faces.shape}')
    return verts, faces


def poisson_mesh_reconstruction(points, normals=None):
    """Poisson surface reconstruction from a point cloud (Open3D backend)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    print(f'[INFO] poisson mesh reconstruction: {points.shape} --> '
          f'{vertices.shape} / {triangles.shape}')
    return vertices, triangles
