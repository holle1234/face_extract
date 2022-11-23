import mediapipe as mp
from cv2 import INTER_CUBIC, INTER_AREA
from cv2 import convexHull, resize, drawContours, bitwise_or
from cv2 import getRotationMatrix2D, warpAffine
from math import sin, cos, atan2, pi, degrees
from numpy import ndarray, clip, hstack, zeros
from numpy import int16, int32, uint8


class FaceAligner:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        self.eyes_l, self.eyes_r = slice(22, 32), slice(252, 262)

    def find_faces(self, img):
        results = self.face_mesh.process(img).multi_face_landmarks
        return results

    def align_face(self, image, size=(160, 160)):
        if not isinstance(image, ndarray):
            return False, None

        result = self.find_faces(image)
        if result:
            # translate landmarks
            x, y = self._to_image_coordinates(result[0].landmark, image.shape[:2])
            # orientation correction
            img, rad, cnt  = self._correct_face_orientation(image, x, y)
            # update coordinates
            x_updated, y_updated = self._update_coordinates(rad, cnt, x, y, img.shape)
            # crop tightly around face
            img = self._crop_face(img, x_updated, y_updated)
            # resize image
            img = self._resize_image(img, size)
            return True, img
        return False, None

    @staticmethod
    def _to_image_coordinates(landmarks, shape):
        # get landmarks and translate coordinates
        h, w = shape
        x = clip([f.x * w for f in landmarks], 0, w).astype(int16)
        y = clip([f.y * h for f in landmarks], 0, h).astype(int16)
        return x, y

    def _correct_face_orientation(self, img, x, y):
        h, w = img.shape[:2]
        # average position of eyes
        xl, xr, yl, yr = self._get_avg_eyes_pos(x, y)
        # calculate orientation correction from eye-level
        rad = self._calculate_orientation_rad(xr, xl, yr, yl)
        # center of image to rotate by
        cnt = (xr + xl) // 2, (yl + yr) // 2
        # rotate image
        angle = -degrees(rad)
        matrix = getRotationMatrix2D(cnt, angle, 1.0)
        rotated = warpAffine(img, matrix, (w, h))
        return rotated, rad, cnt

    def _get_avg_eyes_pos(self, x, y):
        xl, xr = x[self.eyes_l].mean(), x[self.eyes_r].mean()
        yl, yr = y[self.eyes_l].mean(), y[self.eyes_r].mean()
        return xl, xr, yl, yr

    @staticmethod
    def _crop_face(img, x, y):
        h, w = img.shape[:2]
        # reshaping mediapipe facial points for convex hull
        xy = hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        points = xy.reshape((len(x), 1, 2)).astype(int32)
        # create mask to draw points to
        mask = zeros((h, w), dtype=uint8)
        # create hull and draw it
        hull = convexHull(points)
        drawContours(mask, [hull], -1, (255, 0, 0), -1)
        # extract face using bitwise-or
        img = bitwise_or(img, img, mask=mask)
        # crop face tightly from image
        img = img[y.min():y.max(), x.min():x.max()]
        return img

    @staticmethod
    def _calculate_orientation_rad(xr, xl, yr, yl):
        return 2 * pi - atan2(yr - yl, xr - xl)

    @staticmethod
    def _resize_image(img, size=(160, 160)):
        h, w = img.shape[:2]
        if h == w:
            return resize(img, size, INTER_AREA)
        dif = h if h > w else w
        if dif > (size[0] + size[1]) // 2:
            interpolation = INTER_AREA
        else:
            interpolation = INTER_CUBIC
        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2
        mask = zeros((dif, dif, 3), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]
        return resize(mask, size, interpolation)

    @staticmethod
    def _update_coordinates(rad, cnt, x, y, shape):
        h, w = shape[:2]
        _sin, _cos = sin(rad), cos(rad)
        dx, dy = x - cnt[0], y - cnt[1]
        x = clip(cnt[0] + _cos * dx - _sin * dy, 0, w).astype(int16)
        y = clip(cnt[1] + _sin * dx + _cos * dy, 0, h).astype(int16)
        return x, y

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.face_mesh.close()



# from cv2 import VideoCapture, imshow, waitKey
# from cv2 flip, cvtColor, COLOR_BGR2RGB
#
# cap = VideoCapture(0)
# with FaceAligner() as aligner:
#     while True:
#         ret, image = cap.read()
#         image = flip(image, 1)
#         image = cvtColor(image, COLOR_BGR2RGB)
#         _, aligned = aligner.align_face(image)
#
#         if isinstance(aligned, ndarray):
#             aligned = resize(aligned, (250, 250))
#             aligned = cvtColor(aligned, COLOR_BGR2RGB)
#             imshow("img", aligned)
#             waitKey(20)
