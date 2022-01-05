import numpy as np
from multiagent.core import Agent, Landmark


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def triangleHeight(hypotenuse_v, adjacent_v):
    hypotenuse_length = np.linalg.norm(hypotenuse_v)
    theta = angle_between(hypotenuse_v, adjacent_v)
    return hypotenuse_length * np.sin(theta)


def hasLineOfSight(
    source: Agent or Landmark, target: Agent or Landmark, entity: Agent or Landmark
):
    # source to dest vector
    std_v = target.state.p_pos - source.state.p_pos
    # source to entity vextor
    ste_v = entity.state.p_pos - source.state.p_pos

    if triangleHeight(ste_v, std_v) <= entity.size:
        return False

    else:
        return True
