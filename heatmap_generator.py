import numpy as np
import cv2
import os
import sys
sys.path.append('../')


class HeatmapGenerator:
    def __init__(self):
        # Real world pitch dimensions in metres
        # Must match your ViewTransformer court_width and court_length
        self.pitch_width = 68
        self.pitch_length = 52

        # Output heatmap image size in pixels
        self.heatmap_width = 800
        self.heatmap_height = 600

        # Output folder
        self.output_dir = 'output_videos'

    def _collect_positions(self, tracks, team_id):
        """
        Collects all position_transformed coordinates for a given team
        across all frames.
        """
        positions = []
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, player_info in player_track.items():
                if player_info.get('team') != team_id:
                    continue
                pos = player_info.get('position_transformed', None)
                if pos is None:
                    continue
                try:
                    x, y = float(pos[0]), float(pos[1])
                    # Only keep positions within valid pitch bounds
                    if 0 <= x <= self.pitch_length and 0 <= y <= self.pitch_width:
                        positions.append((x, y))
                except (TypeError, IndexError, ValueError):
                    continue
        return positions

    def _positions_to_heatmap(self, positions):
        """
        Converts a list of (x, y) real-world positions into a
        smooth heatmap image using Gaussian accumulation.
        """
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)

        if not positions:
            return heatmap

        for x, y in positions:
            # Convert real-world metres to pixel coordinates
            px = int((x / self.pitch_length) * (self.heatmap_width - 1))
            py = int((y / self.pitch_width) * (self.heatmap_height - 1))

            # Clamp to image bounds
            px = max(0, min(self.heatmap_width - 1, px))
            py = max(0, min(self.heatmap_height - 1, py))

            heatmap[py, px] += 1.0

        # Apply Gaussian blur to smooth the density
        heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)

        # Normalize to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        return heatmap

    def _draw_pitch_lines(self, image):
        """
        Draws basic pitch markings on the heatmap image.
        """
        h, w = image.shape[:2]
        line_color = (255, 255, 255)
        thickness = 2

        # Pitch border
        cv2.rectangle(image, (10, 10), (w - 10, h - 10), line_color, thickness)

        # Halfway line
        cv2.line(image, (w // 2, 10), (w // 2, h - 10), line_color, thickness)

        # Centre circle
        cv2.circle(image, (w // 2, h // 2), h // 8, line_color, thickness)

        # Left penalty box
        box_w = int(w * 0.15)
        box_h = int(h * 0.45)
        box_y = (h - box_h) // 2
        cv2.rectangle(image, (10, box_y), (10 + box_w, box_y + box_h), line_color, thickness)

        # Right penalty box
        cv2.rectangle(image, (w - 10 - box_w, box_y),
                      (w - 10, box_y + box_h), line_color, thickness)

        return image

    def generate_team_heatmap(self, tracks, team_id, team_label="Team"):
        """
        Generates and saves a heatmap for a single team.
        Returns the heatmap image as a numpy array.
        """
        positions = self._collect_positions(tracks, team_id)
        print(f"{team_label}: {len(positions)} position samples collected")

        heatmap_raw = self._positions_to_heatmap(positions)

        # Apply JET colormap — blue=low, red=high density
        heatmap_colored = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

        # Draw pitch lines on top
        heatmap_colored = self._draw_pitch_lines(heatmap_colored)

        # Add team label
        cv2.putText(heatmap_colored, f"{team_label} Position Heatmap",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(heatmap_colored, f"Total samples: {len(positions)}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # Save
        filename = f"{team_label.lower().replace(' ', '_')}_heatmap.png"
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, heatmap_colored)
        print(f"Saved: {save_path}")

        return heatmap_colored

    def generate_combined_heatmap(self, tracks):
        """
        Generates a side-by-side combined heatmap for both teams.
        Saves as combined_heatmap.png and returns the image.
        """
        heatmap_team1 = self.generate_team_heatmap(tracks, team_id=1, team_label="Team 1")
        heatmap_team2 = self.generate_team_heatmap(tracks, team_id=2, team_label="Team 2")

        # Add a divider between the two
        divider = np.zeros((self.heatmap_height, 10, 3), dtype=np.uint8)
        divider[:] = (100, 100, 100)

        combined = np.hstack([heatmap_team1, divider, heatmap_team2])

        # Add title to combined image
        title_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, "Football Match - Player Position Heatmaps",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)

        combined_with_title = np.vstack([title_bar, combined])

        save_path = os.path.join(self.output_dir, 'combined_heatmap.png')
        cv2.imwrite(save_path, combined_with_title)
        print(f"Saved: {save_path}")

        return combined_with_title