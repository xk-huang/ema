# ZJU MoCap Constants

class ZJU_MOCAP_CONSTANTS:
    WIDTH = 1024
    HEIGHT = 1024

    MASK_DIR_NAME_TO_MASK_SUFFIX = {
        "mask_cihp": ".png",
        "mask-schp": "_0.png"
    }


# SMPL Constants
class SMPL_CONSTANTS:
    JOINT_NAMES = [
        "root",
        "lhip",
        "rhip",
        "belly",
        "lknee",
        "rknee",
        "spine",
        "lankle",
        "rankle",
        "chest",
        "ltoes",
        "rtoes",
        "neck",
        "linshoulder",
        "rinshoulder",
        "head",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhand",
        "rhand",
    ]

    BONE_NAMES = [
        ("root", "lhip"),
        ("root", "rhip"),
        ("root", "belly"),
        ("lhip", "lknee"),
        ("rhip", "rknee"),
        ("belly", "spine"),
        ("lknee", "lankle"),
        ("rknee", "rankle"),
        ("spine", "chest"),
        ("lankle", "ltoes"),
        ("rankle", "rtoes"),
        ("chest", "neck"),
        ("chest", "linshoulder"),
        ("chest", "rinshoulder"),
        ("neck", "head"),
        ("linshoulder", "lshoulder"),
        ("rinshoulder", "rshoulder"),
        ("lshoulder", "lelbow"),
        ("rshoulder", "relbow"),
        ("lelbow", "lwrist"),
        ("relbow", "rwrist"),
        ("lwrist", "lhand"),
        ("rwrist", "rhand"),
    ]

    BONE_IDS_TO_UNIQ_TF_IDS = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]

    # BONE_HEAD_JOINT_IDS = [JOINT_NAMES.index(head) for (head, _) in BONE_NAMES]
    BONE_HEAD_JOINT_IDS = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21
    ]
