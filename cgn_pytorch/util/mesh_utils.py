"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import numpy as np
import pickle
from tqdm import tqdm
import trimesh
import trimesh.transformations as tra
from importlib_resources import files

MESH_DIR = str(files("cgn_pytorch").joinpath("gripper/mesh"))


class SceneObject(object):
    """Represents a graspable object."""

    def __init__(self, filename: str):
        """Create a SceneObject object.

        Args:
            filename (str): Path to the mesh file.
        """
        trimesh.util.log.setLevel("ERROR")
        self.mesh = trimesh.load(filename, force="mesh", skip_materials=True)
        self.scale = 1.0

        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("object", self.mesh)

    def rescale(self, scale: int = 1.0):
        """Rescale the object.

        Args:
            scale (float, optional): Defaults to 1.0.
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size: float = 1.0):
        """Set longest of all three lengths in Cartesian space.

        Args:
            size (float, optional). Defaults to 1.0.
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.
        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)


class PandaGripper(object):
    """A SceneObject representing a Franka Panda gripper."""
    DEFAULT_GRIPPER_WIDTH = 0.08
    def __init__(self, q: list[float]=0.04, num_contact_points_per_finger: int = 10):
        """Create a Franka Panda parallel-yaw gripper object.

        Args:
            q (float, optional): Opening Configuration. Defaults to 0.04.
            num_contact_points_per_finger (int, optional). Defaults to 10.
        """
        self.joint_limits = [0.0, 0.04]
        self.root_folder = str(files("cgn_pytorch").joinpath("gripper"))
        if q is None:
            q = self.DEFAULT_GRIPPER_WIDTH
        self.q = q
        fn_base = str(
            files("cgn_pytorch").joinpath("gripper/mesh/panda_gripper/hand.stl")
        )
        fn_finger = str(
            files("cgn_pytorch").joinpath("gripper/mesh/panda_gripper/finger.stl")
        )
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.contact_ray_origins = []
        self.contact_ray_directions = []

        coords_path =  str(
                files("cgn_pytorch").joinpath(
                    "gripper/control_points/panda_gripper_coords.pickle"
                )
            )
        with open(coords_path, 'rb') as f:
            self.finger_coords = pickle.load(f, encoding="latin1")
        finger_direction = (
            self.finger_coords["gripper_right_center_flat"]
            - self.finger_coords["gripper_left_center_flat"]
        )
        self.contact_ray_origins.append(
            np.r_[self.finger_coords["gripper_left_center_flat"], 1]
        )
        self.contact_ray_origins.append(
            np.r_[self.finger_coords["gripper_right_center_flat"], 1]
        )
        self.contact_ray_directions.append(
            finger_direction / np.linalg.norm(finger_direction)
        )
        self.contact_ray_directions.append(
            -finger_direction / np.linalg.norm(finger_direction)
        )

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.
        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays_contact(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.
        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix
            contact_ray_origin {[nump.array]} -- a 4x1 homogeneous vector
            contact_ray_direction {[nump.array]} -- a 4x1 homogeneous vector
        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(self.contact_ray_origins.T).T, transform[
            :3, :3
        ].dot(self.contact_ray_directions.T).T

    def get_control_point_tensor(
        self, batch_size: int, symmetric: bool = False, convex_hull: bool = True
    ) -> np.ndarray:
        """Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        Args:
            batch_size (int): Batch size.
            symmetric (bool, optional): Output the symmetric control point configuration of the gripper Defaults to False.
            convex_hull (bool, optional): Return control points according to the convex hull panda gripper model. Defaults to True.

        Returns:
              np.ndarray -- control points of the panda gripper
        """
        control_points = np.load(
            str(files("cgn_pytorch").joinpath("gripper/control_points/panda.npy")),
            encoding="bytes",
        )[:, :3]
        if symmetric:
            control_points = [
                [0, 0, 0],
                control_points[1, :],
                control_points[0, :],
                control_points[-1, :],
                control_points[-2, :],
            ]
        else:
            control_points = [
                [0, 0, 0],
                control_points[0, :],
                control_points[1, :],
                control_points[-2, :],
                control_points[-1, :],
            ]

        control_points = np.asarray(control_points, dtype=np.float32)
        if not convex_hull:
            # actual depth of the gripper different from convex collision model
            control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])
        return control_points


def create_gripper(
    name: str, configuration: list[float] = None, mesh_folder: str = MESH_DIR
):
    """Create a gripper object.
    Arguments:
        name {str} -- name of the gripper
    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})
    Raises:
        Exception: If the gripper name is unknown.
    Returns:
        [type] -- gripper object
    """
    if name.lower() == "panda":
        return PandaGripper(q=configuration)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(
    object_mesh, gripper_transforms, gripper_name, silent=False
):
    """Check collision of object with gripper.
    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper
    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})
    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object("object", object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(
            np.min(
                [
                    manager.min_distance_single(gripper_mesh, transform=tf)
                    for gripper_mesh in gripper_meshes
                ]
            )
        )

    return [d == 0 for d in min_distance], min_distance


def grasp_contact_location(
    transforms, successfuls, collisions, object_mesh, gripper_name="panda", silent=False
):
    """Computes grasp contacts on objects and normals, offsets, directions
    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh
    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})
    Returns:
        list of dicts of contact information per grasp ray
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True
        )
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    for p, colliding, outcome in tqdm(
        zip(transforms, collisions, successfuls), total=len(transforms), disable=silent
    ):
        contact_dict = {}
        contact_dict["collisions"] = 0
        contact_dict["valid_locations"] = 0
        contact_dict["successful"] = outcome
        contact_dict["grasp_transform"] = p
        contact_dict["contact_points"] = []
        contact_dict["contact_directions"] = []
        contact_dict["contact_face_normals"] = []
        contact_dict["contact_offsets"] = []

        if colliding:
            contact_dict["collisions"] = 1
        else:
            ray_origins, ray_directions = gripper.get_closing_rays_contact(p)

            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False
            )

            if len(locations) > 0:
                valid_locations = (
                    np.linalg.norm(ray_origins[index_ray] - locations, axis=1)
                    <= 2.0 * gripper.q
                )

                if sum(valid_locations) > 1:
                    contact_dict["valid_locations"] = 1
                    contact_dict["contact_points"] = locations[valid_locations]
                    contact_dict["contact_face_normals"] = object_mesh.face_normals[
                        index_tri[valid_locations]
                    ]
                    contact_dict["contact_directions"] = ray_directions[
                        index_ray[valid_locations]
                    ]
                    contact_dict["contact_offsets"] = np.linalg.norm(
                        ray_origins[index_ray[valid_locations]]
                        - locations[valid_locations],
                        axis=1,
                    )
                    res.append(contact_dict)

    return res
