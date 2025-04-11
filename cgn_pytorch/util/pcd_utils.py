from dataclasses import field
from typing import dataclass_transform
import numpy as np
from scipy.spatial.transform import Rotation as R
from pydantic import BaseModel
from mbodios.types.ndarray import array,sz,Float

from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import Field as PydanticField
from pydantic.fields import PrivateAttr as PydanticPrivateAttr
from pydantic.fields import ComputedFieldInfo as PydanticComputedFieldInfo
from dataclasses import field,Field as DataclassField
from pydantic.fields import FieldInfo as PydanticFieldInfo
from pydantic.fields import ModelPrivateAttr as PydanticPrivateFieldInfo


@dataclass_transform(
    kw_only_default=False,
    field_specifiers=(PydanticField,PydanticPrivateAttr,PydanticComputedFieldInfo,DataclassField,field,PydanticFieldInfo,PydanticPrivateFieldInfo)  
)
class ArgsAllowed(ModelMetaclass):...
class Position(BaseModel,metaclass=ArgsAllowed):
    x: float
    y: float
    z: float

class Orientation(BaseModel,metaclass=ArgsAllowed):
    x: float
    y: float
    z: float
    w: float

class Pose(BaseModel,metaclass=ArgsAllowed):
    position: Position
    orientation: Orientation

class Header(BaseModel,metaclass=ArgsAllowed):
    frame_id: str

class PoseStamped(BaseModel,metaclass=ArgsAllowed):
    position: Position
    orientation: Orientation
    pose: Pose
    header: Header

    @classmethod
    def from_matrix(cls,matrix:"array[sz[4],sz[4],Float]",frame_id:str="world") -> "PoseStamped":
        quat = R.from_dcm(matrix[-1, :3, :3]).as_quat()
        trans = matrix[:-1, -1]
        pose = list(trans) + list(quat)
        pose = list2pose_stamped(pose, frame_id=frame_id)
        return pose



def pose_from_matrix(matrix, frame_id="world"):
    quat = R.from_dcm(matrix[-1, :3, :3]).as_quat()
    trans = matrix[:-1, -1]
    pose = list(trans) + list(quat)
    pose = list2pose_stamped(pose, frame_id=frame_id)
    return pose
        
def list2pose_stamped(pose, frame_id="world"):
    msg = PoseStamped(
        position=Position(*pose[:3]),
        orientation=Orientation(*pose[3:]),
        pose=Pose(position=Position(*pose[:3]),orientation=Orientation(*pose[3:])),
        header=Header(frame_id=frame_id)
    )
    return msg
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]
    return msg


def unit_pose():
    return list2pose_stamped([0, 0, 0, 0, 0, 0, 1])

def unit_pose_matrix():
    return np.array([[0, 0, 0],[0, 0, 0],[0, 0, 1]])

def get_transform(pose_frame_target, pose_frame_source):
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = matrix_from_pose(pose_frame_target)
    T_source_world = matrix_from_pose(pose_frame_source)
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = pose_from_matrix(
        T_relative_world, frame_id=pose_frame_source.header.frame_id)
    return pose_relative_world

def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id=None):
    T_pose_source = matrix_from_pose(pose_source)
    pose_transform_target2source = get_transform(
        pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(
        pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target, frame_id=frame_id)
    return pose_target

def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]

def matrix_from_pose(pose: PoseStamped) -> "array[sz[4],sz[4]]":
    pose_list = pose_stamped2list(pose)
    trans = pose_list[0:3]
    quat = pose_list[3:7]

    T = np.zeros((4, 4))
    T[-1, -1] = 1
    r = R.from_quat(quat)
    T[:3, :3] = r.as_dcm()
    T[0:3, 3] = trans
    return T

def farthest_point_downsample(pointcloud: np.ndarray, k: int) -> np.ndarray :
    '''                                                                                                                           
    pointcloud (numpy array): cartesian points, shape (N, 3)                                                                      
    k (int): number of points to sample                                                                                           
                                                                                                                                  
    sampled_cloud (numpy array): downsampled points, shape (k, 3)                                                                 
    '''
    start_ind = np.random.randint(0, len(pointcloud)) # pick a random point in the cloud to start                                 
    sampled_cloud = np.array([pointcloud[start_ind]])
    pointcloud = np.delete(pointcloud, start_ind, axis=0)
    mindists = np.full(len(pointcloud), np.inf) # make a list of minimum distances to samples for each point                      
    for i in range(k):
        last_sample = sampled_cloud[-1]
        ptdists = ((pointcloud-last_sample)**2).sum(axis=1) # distances between each point and most recent sample
        mindists = np.minimum(ptdists, mindists)
        min_ind = np.argmax(mindists)
        sampled_cloud = np.append(sampled_cloud, [pointcloud[min_ind]], axis=0)
        pointcloud = np.delete(pointcloud, min_ind, axis=0)
        mindists = np.delete(mindists, min_ind, axis=0)
    return sampled_cloud
