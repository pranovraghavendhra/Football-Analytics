import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import measure_distance


class PassDetector:
    def __init__(self):
        self.min_possession_frames = 8
        self.min_pass_distance = 1.5
        self.min_frames_between_same_pair = 20
        self.passes = []

    def detect_passes(self, tracks):
        possession_history = []

        for frame_num, player_track in enumerate(tracks['players']):
            possessor = None
            for player_id, player_info in player_track.items():
                if player_info.get('has_ball', False):
                    possessor = {
                        'player_id': player_id,
                        'team': player_info.get('team', -1),
                        'frame': frame_num,
                        'position': player_info.get('position_transformed', None)
                    }
                    break
            possession_history.append(possessor)

        passes = []
        current_holder = None
        current_holder_since = 0
        recent_pairs = {}
        loose_frames = 0          # declared OUTSIDE the loop
        MAX_LOOSE_FRAMES = 45
        MAX_VALID_PLAYER_ID = 50

        # Single loop — no nesting
        for frame_num, possessor in enumerate(possession_history):

            if possessor is None:
                loose_frames += 1
                if loose_frames > MAX_LOOSE_FRAMES:
                    current_holder = None
                    current_holder_since = frame_num
                continue

            # Ball is with someone — reset loose counter
            loose_frames = 0

            if current_holder is None:
                current_holder = possessor
                current_holder_since = frame_num
                continue

            same_player = (possessor['player_id'] == current_holder['player_id'])

            if not same_player:
                # Filter ghost IDs
                if (current_holder['player_id'] > MAX_VALID_PLAYER_ID or
                        possessor['player_id'] > MAX_VALID_PLAYER_ID):
                    current_holder = possessor
                    current_holder_since = frame_num
                    continue

                frames_held = frame_num - current_holder_since

                if frames_held >= self.min_possession_frames:
                    same_team = (possessor['team'] == current_holder['team'])

                    distance = None
                    if (current_holder['position'] is not None and
                            possessor['position'] is not None):
                        try:
                            distance = measure_distance(
                                current_holder['position'],
                                possessor['position']
                            )
                            distance = round(distance, 2)
                        except Exception:
                            distance = None

                    if distance is not None and distance < self.min_pass_distance:
                        current_holder = possessor
                        current_holder_since = frame_num
                        continue

                    pair_key = (current_holder['player_id'], possessor['player_id'])
                    reverse_key = (possessor['player_id'], current_holder['player_id'])
                    last_frame = max(
                        recent_pairs.get(pair_key, -999),
                        recent_pairs.get(reverse_key, -999)
                    )
                    if frame_num - last_frame < self.min_frames_between_same_pair:
                        current_holder = possessor
                        current_holder_since = frame_num
                        continue

                    pass_event = {
                        'frame': frame_num,
                        'passer_id': current_holder['player_id'],
                        'receiver_id': possessor['player_id'],
                        'passer_team': current_holder['team'],
                        'receiver_team': possessor['team'],
                        'type': 'pass' if same_team else 'interception',
                        'passer_position': current_holder['position'],
                        'receiver_position': possessor['position'],
                        'distance_meters': distance,
                    }
                    passes.append(pass_event)
                    recent_pairs[pair_key] = frame_num

                current_holder = possessor
                current_holder_since = frame_num

        self.passes = passes
        return passes

    def get_pass_summary(self):
        if not self.passes:
            return {
                'total_passes': 0,
                'team1_passes': 0,
                'team2_passes': 0,
                'interceptions': 0,
                'player_pass_count': {},
                'avg_pass_distance_meters': None,
                'passes_outside_polygon': 0,
                'all_passes': []
            }

        team1_passes = [p for p in self.passes
                        if p['type'] == 'pass' and p['passer_team'] == 1]
        team2_passes = [p for p in self.passes
                        if p['type'] == 'pass' and p['passer_team'] == 2]
        interceptions = [p for p in self.passes if p['type'] == 'interception']

        player_pass_count = {}
        for p in self.passes:
            if p['type'] == 'pass':
                pid = p['passer_id']
                player_pass_count[pid] = player_pass_count.get(pid, 0) + 1

        distances = [p['distance_meters'] for p in self.passes
                     if p['distance_meters'] is not None]
        avg_distance = round(sum(distances) / len(distances), 2) if distances else None
        no_distance_count = sum(1 for p in self.passes
                                if p['distance_meters'] is None)

        return {
            'total_passes': len(self.passes),
            'team1_passes': len(team1_passes),
            'team2_passes': len(team2_passes),
            'interceptions': len(interceptions),
            'player_pass_count': player_pass_count,
            'avg_pass_distance_meters': avg_distance,
            'passes_outside_polygon': no_distance_count,
            'all_passes': self.passes
        }

    def draw_pass_annotations(self, video_frames, tracks):
        pass_by_frame = {}
        for p in self.passes:
            pass_by_frame[p['frame']] = p

        ANNOTATION_DURATION = 30
        output_frames = []
        active_annotations = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num in pass_by_frame:
                event = pass_by_frame[frame_num]
                active_annotations.append((event, frame_num + ANNOTATION_DURATION))

            active_annotations = [
                (ev, exp) for ev, exp in active_annotations if frame_num <= exp
            ]

            for event, expiry in active_annotations:
                self._draw_pass_event(frame, event, tracks, frame_num)

            frame = self._draw_pass_counter(frame, frame_num)
            output_frames.append(frame)

        return output_frames

    def _draw_pass_event(self, frame, event, tracks, frame_num):
        passer_id   = event['passer_id']
        receiver_id = event['receiver_id']
        is_pass     = event['type'] == 'pass'
        color       = (0, 200, 80) if is_pass else (0, 60, 255)
        label       = "PASS" if is_pass else "INTERCEPTION"

        player_track  = tracks['players'][frame_num]
        passer_bbox   = player_track.get(passer_id, {}).get('bbox', None)
        receiver_bbox = player_track.get(receiver_id, {}).get('bbox', None)

        if passer_bbox is None:
            for offset in range(1, 15):
                f = max(0, frame_num - offset)
                passer_bbox = tracks['players'][f].get(
                    passer_id, {}).get('bbox', None)
                if passer_bbox:
                    break

        if receiver_bbox is None:
            for offset in range(1, 15):
                f = min(len(tracks['players']) - 1, frame_num + offset)
                receiver_bbox = tracks['players'][f].get(
                    receiver_id, {}).get('bbox', None)
                if receiver_bbox:
                    break

        if passer_bbox and receiver_bbox:
            px = int((passer_bbox[0] + passer_bbox[2]) / 2)
            py = int(passer_bbox[3])
            rx = int((receiver_bbox[0] + receiver_bbox[2]) / 2)
            ry = int(receiver_bbox[3])

            cv2.line(frame, (px, py), (rx, ry), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.circle(frame, (rx, ry), 6, color, -1)

            mid_x = (px + rx) // 2
            mid_y = (py + ry) // 2
            dist_text = f" {event['distance_meters']}m" \
                if event['distance_meters'] else ""
            cv2.putText(frame, f"{label}{dist_text}",
                        (mid_x - 40, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_pass_counter(self, frame, frame_num):
        passes_so_far = [p for p in self.passes if p['frame'] <= frame_num]
        team1     = sum(1 for p in passes_so_far
                        if p['type'] == 'pass' and p['passer_team'] == 1)
        team2     = sum(1 for p in passes_so_far
                        if p['type'] == 'pass' and p['passer_team'] == 2)
        intercepts = sum(1 for p in passes_so_far
                         if p['type'] == 'interception')

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 860), (380, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"Team 1 Passes: {team1}", (10, 895),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Passes: {team2}", (10, 930),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Interceptions: {intercepts}", (10, 965),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame