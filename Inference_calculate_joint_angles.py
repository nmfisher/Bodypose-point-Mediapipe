import argparse
import mediapipe as mp
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def convert_to_dictionary(kpts):
    # its easier to manipulate keypoints by joint name
    keypoints_to_index = {'lefthip': 6, 'leftknee': 8, 'leftfoot': 10,
                          'righthip': 7, 'rightknee': 9, 'rightfoot': 11,
                          'leftshoulder': 0, 'leftelbow': 2, 'leftwrist': 4,
                          'rightshoulder': 1, 'rightelbow': 3, 'rightwrist': 5,
                          'left_heal' : 12, 'right_heal' : 13, 'left_foot_index' : 14, 'right_foot_index'  : 15}

    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        kpts_dict[key] = kpts[:, k_index]

    kpts_dict['joints'] = list(keypoints_to_index.keys())

    return kpts_dict


def add_hips_and_neck(kpts):
    # we add two new keypoints which are the mid point between the hips and mid point between the shoulders

    # add hips kpts
    difference = kpts['lefthip'] - kpts['righthip']
    difference = difference/2
    hips = kpts['righthip'] + difference
    kpts['hips'] = hips
    kpts['joints'].append('hips')

    # add neck kpts
    difference = kpts['leftshoulder'] - kpts['rightshoulder']
    difference = difference/2
    neck = kpts['rightshoulder'] + difference
    kpts['neck'] = neck
    kpts['joints'].append('neck')

    hierarchy = {'hips': [],
                 'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
                 'left_heal': ['leftfoot', 'leftknee', 'lefthip', 'hips'],
                 'left_foot_index': ['leftfoot', 'leftknee', 'lefthip', 'hips'],

                 'righthip': ['hips'], 'rightknee': ['righthip', 'hips'],
                 'rightfoot': ['rightknee', 'righthip', 'hips'],
                 'right_heal': ['rightfoot', 'rightknee', 'righthip', 'hips'],
                 'right_foot_index': ['rightfoot', 'rightknee', 'righthip', 'hips'],
                 'neck': ['hips'],
                 'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'],
                 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
                 'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'],
                 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
                 }
    kpts['hierarchy'] = hierarchy
    kpts['root_joint'] = 'hips'

    return kpts

# remove jittery keypoints by applying a median filter along each axis


def median_filter(kpts, window_size=3):

    import copy
    filtered = copy.deepcopy(kpts)

    from scipy.signal import medfilt

    # apply median filter to get rid of poor keypoints estimations
    for joint in filtered['joints']:
        joint_kpts = filtered[joint]
        xs = joint_kpts[:, 0]
        ys = joint_kpts[:, 1]
        zs = joint_kpts[:, 2]
        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)
        filtered[joint] = np.stack([xs, ys, zs], axis=-1)

    return filtered


def get_bone_lengths(kpts):
    """
    We have to define an initial skeleton pose(T pose).
    In this case we need to known the length of each bone.
    Here we calculate the length of each bone from data
    """

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips':
            continue
        parent = kpts['hierarchy'][joint][0]

        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        _bone = joint_kpts - parent_kpts
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis=-1))

        _bone_length = np.median(_bone_lengths)
        bone_lengths[joint] = _bone_length

        # plt.hist(bone_lengths, bins = 25)
        # plt.title(joint)
        # plt.show()

    # print(bone_lengths)
    kpts['bone_lengths'] = bone_lengths
    return

# Here we define the T pose and we normalize the T pose by the length of the hips to neck distance.


def get_base_skeleton(kpts, normalization_bone='neck'):

    # this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts['bone_lengths']

    # define skeleton offset directions
    offset_directions = {}
    offset_directions['lefthip'] = np.array([1, 0, 0])
    offset_directions['leftknee'] = np.array([0, -1, 0])
    offset_directions['leftfoot'] = np.array([0, -1, 0])
    offset_directions['left_heal'] = np.array([0, -1, 0])
    offset_directions['left_foot_index'] = np.array([0, -1, 0])

    offset_directions['righthip'] = np.array([-1, 0, 0])
    offset_directions['rightknee'] = np.array([0, -1, 0])
    offset_directions['rightfoot'] = np.array([0, -1, 0])
    offset_directions['right_heal'] = np.array([0, -1, 0])
    offset_directions['right_foot_index'] = np.array([0, -1, 0])

    offset_directions['neck'] = np.array([0, 1, 0])

    offset_directions['leftshoulder'] = np.array([1, 0, 0])
    offset_directions['leftelbow'] = np.array([1, 0, 0])
    offset_directions['leftwrist'] = np.array([1, 0, 0])

    offset_directions['rightshoulder'] = np.array([-1, 0, 0])
    offset_directions['rightelbow'] = np.array([-1, 0, 0])
    offset_directions['rightwrist'] = np.array([-1, 0, 0])

    # set bone normalization length. Set to 1 if you dont want normalization
    normalization = kpts['bone_lengths'][normalization_bone]
    # normalization = 1

    # base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
    base_skeleton = {'hips': np.array([0, 0, 0])}

    def _set_length(joint_type):
        base_skeleton['left' + joint_type] = offset_directions['left' + joint_type] * \
            ((body_lengths['left' + joint_type] +
             body_lengths['right' + joint_type])/(2 * normalization))
        base_skeleton['right' + joint_type] = offset_directions['right' + joint_type] * \
            ((body_lengths['left' + joint_type] +
             body_lengths['right' + joint_type])/(2 * normalization))


    _set_length('hip')
    _set_length('knee')
    _set_length('foot')
    _set_length('shoulder')
    _set_length('elbow')
    _set_length('wrist')
    _set_length('_heal')
    _set_length('_foot_index')
    base_skeleton['neck'] = offset_directions['neck'] * \
        (body_lengths['neck']/normalization)

    kpts['offset_directions'] = offset_directions
    kpts['base_skeleton'] = base_skeleton
    kpts['normalization'] = normalization

    return

# calculate the rotation of the root joint with respect to the world coordinates


def get_hips_position_and_rotation(frame_pos, root_joint='hips', root_define_joints=['lefthip', 'neck']):

    # root position is saved directly
    root_position = frame_pos[root_joint]

    # calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u/np.sqrt(np.sum(np.square(root_u)))
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v/np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    # Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz, thetay, thetax = utils.Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation

# calculate the rotation matrix and joint angles input joint


def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):

    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0:
            continue
        _r_angles = frame_rotations[parent_name]
        R = utils.get_R_z(
            _r_angles[0]) @ utils.get_R_x(_r_angles[1]) @ utils.get_R_y(_r_angles[2])
        _invR = _invR@R.T

    b = _invR @ (frame_pos[joint_name] -
                 frame_pos[joints_hierarchy[joint_name][0]])

    _R = utils.Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = utils.Decompose_R_ZXY(_R)
    joint_rs = np.array([tz, tx, ty])
    # print(np.degrees(joint_rs))

    return joint_rs

# helper function that composes a chain of rotation matrices


def get_rotation_chain(joint, hierarchy, frame_rotations):

    hierarchy = hierarchy[::-1]

    # this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy:
        angles = frame_rotations[parent]
        _R = utils.get_R_z(
            angles[0])@utils.get_R_x(angles[1])@utils.get_R_y(angles[2])
        R = R @ _R

    return R

# calculate the joint angles frame by frame.


def calculate_joint_angles(kpts):

    # set up emtpy container for joint angles
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = []

    for framenum in range(kpts['hips'].shape[0]):

        # get the keypoints positions in the current frame
        frame_pos = {}
        for joint in kpts['joints']:
            frame_pos[joint] = kpts[joint][framenum]

        root_position, root_rotation = get_hips_position_and_rotation(
            frame_pos)

        frame_rotations = {'hips': root_rotation}

        # center the body pose
        for joint in kpts['joints']:
            frame_pos[joint] = frame_pos[joint] - root_position

        # get the max joints connectsion
        max_connected_joints = 0
        for joint in kpts['joints']:
            if len(kpts['hierarchy'][joint]) > max_connected_joints:
                max_connected_joints = len(kpts['hierarchy'][joint])

        depth = 2
        while (depth <= max_connected_joints):
            for joint in kpts['joints']:
                if len(kpts['hierarchy'][joint]) == depth:
                    joint_rs = get_joint_rotations(
                        joint, kpts['hierarchy'], kpts['offset_directions'], frame_rotations, frame_pos)
                    parent = kpts['hierarchy'][joint][0]
                    frame_rotations[parent] = joint_rs
            depth += 1

        # for completeness, add zero rotation angles for endpoints. This is not necessary as they are never used.
        for _j in kpts['joints']:
            if _j not in list(frame_rotations.keys()):
                frame_rotations[_j] = np.array([0., 0., 0.])

        # update dictionary with current angles.
        for joint in kpts['joints']:
            kpts[joint + '_angles'].append(frame_rotations[joint])

    # convert joint angles list to numpy arrays.
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = np.array(kpts[joint + '_angles'])
        # print(joint, kpts[joint+'_angles'].shape)

    return

# recalculate joint positions from calculated joint angles and draw


def draw_skeleton_from_joint_angles(kpts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum in range(kpts['hips'].shape[0]):

        # get a dictionary containing the rotations for the current frame
        frame_rotations = {}
        for joint in kpts['joints']:
            frame_rotations[joint] = kpts[joint+'_angles'][framenum]

        # for plotting
        for _j in kpts['joints']:
            if _j == 'hips':
                continue

            # get hierarchy of how the joint connects back to root joint
            hierarchy = kpts['hierarchy'][_j]

            # get the current position of the parent joint
            r1 = kpts['hips'][framenum]/kpts['normalization']
            for parent in hierarchy:
                if parent == 'hips':
                    continue
                R = get_rotation_chain(
                    parent, kpts['hierarchy'][parent], frame_rotations)
                r1 = r1 + R @ kpts['base_skeleton'][parent]

            # get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
            r2 = r1 + \
                get_rotation_chain(
                    hierarchy[0], hierarchy, frame_rotations) @ kpts['base_skeleton'][_j]
            plt.plot(xs=[r1[0], r2[0]], ys=[r1[1], r2[1]],
                     zs=[r1[2], r2[2]], color='red')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.azim = 90
        ax.elev = -85
        ax.set_title('Pose from joint angles')
        ax.set_xlim3d(-4, 4)
        ax.set_xlabel('x')
        ax.set_ylim3d(-4, 4)
        ax.set_ylabel('y')
        ax.set_zlim3d(-4, 4)
        ax.set_zlabel('z')
        plt.pause(1/30)
        ax.cla()
    plt.close()

if __name__ == '__main__':

    # get video_path from command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        default='cam0_test.mp4')
    args = parser.parse_args()

    video_path = args.video_path
    print(video_path)

    all_points = []

    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
#            image.flags.writeable = True
#            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#            mp_drawing.draw_landmarks(
#                image,
#                results.pose_landmarks,
#                mp_pose.POSE_CONNECTIONS,
#                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

#            print(results.pose_world_landmarks)

            if results.pose_world_landmarks is not None:
                a, b, c = -1, -1, -1
                points = [[a*results.pose_world_landmarks.landmark[11].y, b*results.pose_world_landmarks.landmark[11].x, -results.pose_world_landmarks.landmark[11].z],
                          [a*results.pose_world_landmarks.landmark[12].y, b*results.pose_world_landmarks.landmark[12].x,
                           c*results.pose_world_landmarks.landmark[12].z],
                          [a*results.pose_world_landmarks.landmark[13].y, b*results.pose_world_landmarks.landmark[13].x,
                           c*results.pose_world_landmarks.landmark[13].z],
                          [a*results.pose_world_landmarks.landmark[14].y, b*results.pose_world_landmarks.landmark[14].x,
                           c*results.pose_world_landmarks.landmark[14].z],
                          [a*results.pose_world_landmarks.landmark[15].y, b*results.pose_world_landmarks.landmark[15].x,
                           c*results.pose_world_landmarks.landmark[15].z],
                          [a*results.pose_world_landmarks.landmark[16].y, b*results.pose_world_landmarks.landmark[16].x,
                           c*results.pose_world_landmarks.landmark[16].z],
                          [a*results.pose_world_landmarks.landmark[23].y, b*results.pose_world_landmarks.landmark[23].x,
                           c*results.pose_world_landmarks.landmark[23].z],
                          [a*results.pose_world_landmarks.landmark[24].y, b*results.pose_world_landmarks.landmark[24].x,
                           c*results.pose_world_landmarks.landmark[24].z],
                          [a*results.pose_world_landmarks.landmark[25].y, b*results.pose_world_landmarks.landmark[25].x,
                           c*results.pose_world_landmarks.landmark[25].z],
                          [a*results.pose_world_landmarks.landmark[26].y, b*results.pose_world_landmarks.landmark[26].x,
                           c*results.pose_world_landmarks.landmark[26].z],
                          [a*results.pose_world_landmarks.landmark[27].y, b*results.pose_world_landmarks.landmark[27].x,
                           c*results.pose_world_landmarks.landmark[27].z],
                          [a*results.pose_world_landmarks.landmark[28].y, b*results.pose_world_landmarks.landmark[28].x,
                           c*results.pose_world_landmarks.landmark[28].z],
                          [a * results.pose_world_landmarks.landmark[29].y, b * results.pose_world_landmarks.landmark[29].x,
                           c * results.pose_world_landmarks.landmark[29].z],
                          [a * results.pose_world_landmarks.landmark[30].y, b * results.pose_world_landmarks.landmark[30].x,
                           c * results.pose_world_landmarks.landmark[30].z],
                          [a * results.pose_world_landmarks.landmark[31].y, b * results.pose_world_landmarks.landmark[31].x,
                           c * results.pose_world_landmarks.landmark[31].z],
                          [a * results.pose_world_landmarks.landmark[32].y, b * results.pose_world_landmarks.landmark[32].x,
                           c * results.pose_world_landmarks.landmark[32].z]




                          ]

                all_points.append(points)

            # Flip the image horizontally for a selfie-view display.

 #           cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
 #           if cv2.waitKey(5) & 0xFF == 27:
 #               break

    cap.release()
    all_points = np.array(all_points)

    kpts = all_points

    # rotate to orient the pose better
    R = utils.get_R_z(np.pi/2)
    for framenum in range(kpts.shape[0]):
        for kpt_num in range(kpts.shape[1]):
            kpts[framenum, kpt_num] = R @ kpts[framenum, kpt_num]

    kpts = convert_to_dictionary(kpts)
    add_hips_and_neck(kpts)
    filtered_kpts = median_filter(kpts)
    get_bone_lengths(filtered_kpts)
    get_base_skeleton(filtered_kpts)

    calculate_joint_angles(filtered_kpts)
    # draw_skeleton_from_joint_coordinates(filtered_kpts)
#    draw_skeleton_from_joint_angles(filtered_kpts)
    #print(filtered_kpts)


    def add_bone_definition(indent, joint_name, offsets, hierarchy, bvh_string):
        offsets[joint_name] = [0.0, 0.0, 0.0]
        offset = offsets[joint_name]

        joint_string = f"""
JOINT {joint_name}
{{
	OFFSET {offset[0]} {offset[1]} {offset[2]}
	CHANNELS 3 Xrotation Yrotation Zrotation"""

        bvh_string += "\n".join([("\t" * indent) + line for line in joint_string.split("\n")])

        if isinstance(hierarchy, dict):
            children = hierarchy[joint_name]
            for child in children:
                bvh_string = add_bone_definition(indent + 1, child, offsets, hierarchy[joint_name], bvh_string)  
            joint_string = ""
        else:
            joint_string = f"""
	End Site
	{{
		OFFSET 1.000000 0.000000 0.000000
	}}"""
        joint_string += """
}""" 

        bvh_string += "\n".join([("\t" * indent) + line for line in joint_string.split("\n")])
        return bvh_string 

    bvh_string='''HIERARCHY
ROOT hips
{
	OFFSET 0.000000 0.000000 0.000000
	CHANNELS Xposition Yposition Zposition Xrotation Yrotation Zrotation'''

    hierarchy = {
        'lefthip': { 'leftknee':{ 'leftfoot': {'left_heal', 'left_foot_index'}}},
        'righthip': {'rightknee': {'rightfoot':{'right_heal','right_foot_index'}}},
        'neck': {
            'leftshoulder':{'leftelbow':{'leftwrist'}},
            'rightshoulder':{'rightelbow':{'rightwrist'}},
         }
    }
    indent = 1
    offsets = {}
    for joint in hierarchy:
        bvh_string = add_bone_definition(indent, joint, offsets, hierarchy, bvh_string)
    bvh_string += """
}"""
    print(bvh_string)

