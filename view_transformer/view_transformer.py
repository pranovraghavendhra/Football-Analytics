import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        # Real world dimensions of the VISIBLE pitch area in metres
        # Camera shows roughly half pitch lengthwise (~52m) and full width (68m)
        court_width = 68      # full pitch width — always 68m
        court_length = 52     # visible length — half pitch approx

        # Pixel coordinates of the 4 visible pitch corners in your video
        # Identified from your sample frame (1920x1080)
        # Order: bottom-left, top-left, top-right, bottom-right
        self.pixel_vertices = np.array([
            [0,    755],    # bottom-left (left touchline, bottom)
            [80,   215],    # top-left (left touchline, top — ad board level)
            [1900, 215],    # top-right (right side, top)
            [1920, 755],    # bottom-right (right side, bottom)
        ])

        self.target_vertices = np.array([
            [0,           court_width],   # bottom-left in metres
            [0,           0          ],   # top-left
            [court_length, 0         ],   # top-right
            [court_length, court_width],  # bottom-right
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(
            reshaped_point, self.persepctive_trasnformer)
        return tranform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed