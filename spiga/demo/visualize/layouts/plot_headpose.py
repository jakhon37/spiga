import numpy as np
import cv2

# Demo libs
from spiga.demo.visualize.layouts.plot_basics import BasicLayout


class HeadposeLayout(BasicLayout):

    BasicLayout.thickness_dft['hpose'] = 2

    def __init__(self):
        super().__init__()
        self.hpose_axe_length = 2
        self.focal_ratio = 1

    def draw_headpose(self, canvas, bbox, rot, trl, euler=False, len_axe=None, thick=None,
                      colors=(BasicLayout.colors['blue'], BasicLayout.colors['green'], BasicLayout.colors['red'])):

        trl = np.float32(trl)
        rot = np.float32(rot)
        K = self._camera_matrix(bbox)

        # Init variables if need it
        if len_axe is None:
            len_axe = self.hpose_axe_length
        if thick is None:
            thick = self.thickness['hpose']

        if euler:
            rot = self._euler_to_rotation_matrix(rot)
        print('rot: ', rot)
        
        from scipy.spatial.transform import Rotation as R

        rot2 = [[-2.07666166e-02, -9.99717791e-01,  1.15362764e-02],
            [2.95717962e-04, 1.15326222e-02, 9.99933453e-01],
            [-9.99784307e-01, 2.07686461e-02, 5.61409637e-05]]

        rotation = R.from_matrix(rot)
        euler_angles = rotation.as_euler('xyz', degrees=True)

        pitch_deg, yaw_deg, roll_deg = euler_angles
        print('pitch, yaw, roll: ', pitch_deg, yaw_deg, roll_deg)
        
        import math
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)

        print("pitch in radians:", pitch_rad)
        print("yaw in radians:", yaw_rad)
        print("roll in radians:", roll_rad)
        head_pose = pitch_rad, yaw_rad, roll_rad
        
        
        rotV, _ = cv2.Rodrigues(rot)
        print('rotV: ', rotV)
        points = np.float32([[len_axe, 0, 0], [0, -len_axe, 0], [0, 0, -len_axe], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rotV, trl, K, (0, 0, 0, 0))
        print('axisPoints: ', axisPoints)
        canvas = cv2.line(canvas, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[2].ravel().astype(int)), colors[0], thick)
        canvas = cv2.line(canvas, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[1].ravel().astype(int)), colors[1], thick)
        canvas = cv2.line(canvas, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[0].ravel().astype(int)), colors[2], thick)
        return canvas, head_pose

    @staticmethod
    def _euler_to_rotation_matrix(headpose):
        # http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
        # Change coordinates system
        euler = np.array([-(headpose[0] - 90), -headpose[1], -(headpose[2] + 90)])
        # Convert to radians
        rad = euler * (np.pi / 180.0)
        cy = np.cos(rad[0])
        sy = np.sin(rad[0])
        cp = np.cos(rad[1])
        sp = np.sin(rad[1])
        cr = np.cos(rad[2])
        sr = np.sin(rad[2])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
        Rp = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
        Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
        return np.matmul(np.matmul(Ry, Rp), Rr)

    def _camera_matrix(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        w = x2-x1
        h = y2-y1
        focal_length_x = w * self.focal_ratio
        focal_length_y = h * self.focal_ratio
        face_center = (x1 + (w * 0.5)), (y1 + (h * 0.5))

        cam_matrix = np.array([[focal_length_x, 0, face_center[0]],
                               [0, focal_length_y, face_center[1]],
                               [0, 0, 1]], dtype=np.float32)
        return cam_matrix