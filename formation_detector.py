import numpy as np
import cv2
import sys
sys.path.append('../')

VALID_FORMATIONS = [
    '4-3-3', '4-4-2', '4-2-3-1', '3-5-2',
    '3-4-3', '5-3-2', '5-4-1', '4-5-1',
    '4-1-4-1', '4-3-2-1'
]


class FormationDetector:
    def __init__(self):
        self.sample_every_n_frames = 10
        self.min_players_required = 7
        self.max_valid_player_id = 50
        self.formations = {1: [], 2: []}
        self.dominant_formation = {1: "Unknown", 2: "Unknown"}

    def _get_team_positions(self, player_track, team_id):
        positions = []
        player_ids = []

        for player_id, player_info in player_track.items():
            if player_id > self.max_valid_player_id:
                continue
            if player_info.get('team') != team_id:
                continue

            pos = player_info.get('position_transformed', None)

            if pos is None:
                bbox = player_info.get('bbox', None)
                if bbox is None:
                    continue
                px = (bbox[0] + bbox[2]) / 2
                py = bbox[3]
                pos = [px, py]

            try:
                x, y = float(pos[0]), float(pos[1])
                positions.append([x, y])
                player_ids.append(player_id)
            except (TypeError, IndexError, ValueError):
                continue

        return np.array(positions), player_ids

    def _snap_to_valid_formation(self, lines):
        if not lines:
            return '4-3-3'

        n_lines = len(lines)

        candidates = [f for f in VALID_FORMATIONS
                      if len(f.split('-')) == n_lines]

        if not candidates:
            candidates = VALID_FORMATIONS

        best = '4-3-3'
        best_score = float('inf')

        for candidate in candidates:
            cand_lines = [int(x) for x in candidate.split('-')]
            max_len = max(len(lines), len(cand_lines))
            a = lines + [0] * (max_len - len(lines))
            b = cand_lines + [0] * (max_len - len(cand_lines))
            score = sum(abs(x - y) for x, y in zip(a, b))
            if score < best_score:
                best_score = score
                best = candidate

        return best

    def _positions_to_formation(self, positions):
        n_players = len(positions)
        if n_players < self.min_players_required:
            return None

        if n_players > 11:
            centre_x = np.median(positions[:, 0])
            distances = np.abs(positions[:, 0] - centre_x)
            keep_indices = np.argsort(distances)[:11]
            positions = positions[keep_indices]

        sorted_x = np.sort(positions[:, 0])
        gaps = np.diff(sorted_x)

        final_lines = None

        for n_lines in [4, 3]:
            n_boundaries = n_lines - 1
            boundary_indices = np.argsort(gaps)[::-1][:n_boundaries]
            boundary_indices = np.sort(boundary_indices)

            lines = []
            prev_idx = 0
            for boundary in boundary_indices:
                line = sorted_x[prev_idx:boundary + 1]
                if len(line) > 0:
                    lines.append(len(line))
                prev_idx = boundary + 1
            last_line = sorted_x[prev_idx:]
            if len(last_line) > 0:
                lines.append(len(last_line))

            if not lines:
                continue

            if lines[0] == 1 and len(lines) > 1:
                lines = lines[1:]

            lines = [l for l in lines if l > 0]
            formation_string = "-".join(str(c) for c in lines)

            if formation_string in VALID_FORMATIONS:
                return formation_string

            if final_lines is None:
                final_lines = lines

        return self._snap_to_valid_formation(final_lines if final_lines else [4, 3, 3])

    def _smooth_formations(self, formation_history, window=5):
        if len(formation_history) <= window:
            return formation_history

        smoothed = []
        for i in range(len(formation_history)):
            start = max(0, i - window // 2)
            end = min(len(formation_history), i + window // 2 + 1)
            window_formations = [f['formation'] for f in formation_history[start:end]]
            majority = max(set(window_formations), key=window_formations.count)
            smoothed.append({
                'frame': formation_history[i]['frame'],
                'formation': majority
            })
        return smoothed

    def detect_formations(self, tracks):
        formation_history = {1: [], 2: []}
        total_frames = len(tracks['players'])

        for frame_num in range(0, total_frames, self.sample_every_n_frames):
            player_track = tracks['players'][frame_num]

            for team_id in [1, 2]:
                positions, _ = self._get_team_positions(player_track, team_id)

                if len(positions) < self.min_players_required:
                    continue

                formation = self._positions_to_formation(positions)

                if formation:
                    formation_history[team_id].append({
                        'frame': frame_num,
                        'formation': formation
                    })

        for team_id in [1, 2]:
            formation_history[team_id] = self._smooth_formations(
                formation_history[team_id])

        self.formations = formation_history

        for team_id in [1, 2]:
            if formation_history[team_id]:
                all_formations = [f['formation'] for f in formation_history[team_id]]
                self.dominant_formation[team_id] = max(
                    set(all_formations), key=all_formations.count
                )

        return formation_history

    def get_formation_summary(self):
        summary = {}

        for team_id in [1, 2]:
            history = self.formations[team_id]
            if not history:
                summary[team_id] = {
                    'dominant_formation': 'Unknown',
                    'formation_counts': {},
                    'total_samples': 0
                }
                continue

            all_formations = [f['formation'] for f in history]
            formation_counts = {}
            for f in all_formations:
                formation_counts[f] = formation_counts.get(f, 0) + 1

            formation_counts = dict(
                sorted(formation_counts.items(),
                       key=lambda x: x[1], reverse=True)
            )

            summary[team_id] = {
                'dominant_formation': self.dominant_formation[team_id],
                'formation_counts': formation_counts,
                'total_samples': len(history)
            }

        return summary

    def _parse_formation(self, formation_string):
        try:
            return [int(x) for x in formation_string.split('-')]
        except Exception:
            return [4, 3, 3]

    def _draw_single_lineup(self, formation_string, team_label,
                         dot_color, pitch_w, pitch_h,
                         attacking_direction='right'):
        pitch = np.zeros((pitch_h, pitch_w, 3), dtype=np.uint8)
        pitch[:] = (34, 139, 34)

        white = (255, 255, 255)

    # Pitch border
        cv2.rectangle(pitch, (15, 15), (pitch_w - 15, pitch_h - 15), white, 2)

    # Halfway line indicator (thin line on attacking side)
        if attacking_direction == 'right':
            cv2.line(pitch, (pitch_w - 15, 15),
                     (pitch_w - 15, pitch_h - 15), white, 3)
        # Subtle pitch stripes
            for i in range(5):
                x1 = 15 + i * ((pitch_w - 30) // 5)
                x2 = x1 + ((pitch_w - 30) // 10)
                cv2.rectangle(pitch, (x1, 15), (x2, pitch_h - 15),
                              (30, 130, 30), -1)
        else:
            cv2.line(pitch, (15, 15),
                     (15, pitch_h - 15), white, 3)
            for i in range(5):
                x1 = 15 + i * ((pitch_w - 30) // 5)
                x2 = x1 + ((pitch_w - 30) // 10)
                cv2.rectangle(pitch, (x1, 15), (x2, pitch_h - 15),
                              (30, 130, 30), -1)

    # Redraw border on top of stripes
        cv2.rectangle(pitch, (15, 15), (pitch_w - 15, pitch_h - 15), white, 2)

    # Penalty box
        box_w = int(pitch_w * 0.18)
        box_h = int(pitch_h * 0.5)
        box_y = (pitch_h - box_h) // 2
        if attacking_direction == 'right':
            cv2.rectangle(pitch, (pitch_w - 15 - box_w, box_y),
                          (pitch_w - 15, box_y + box_h), white, 1)
        else:
           cv2.rectangle(pitch, (15, box_y),
                          (15 + box_w, box_y + box_h), white, 1)

        lines = self._parse_formation(formation_string)
        all_lines = [1] + lines  # GK + outfield lines

        n_lines = len(all_lines)

    # Tighter margins so dots don't overlap labels
        margin_x = 70
        margin_y = 70       # top and bottom margin — keeps dots away from labels
        usable_w = pitch_w - 2 * margin_x
        usable_h = pitch_h - 2 * margin_y
    
        if attacking_direction == 'right':
            x_positions = [
                margin_x + int((i / (n_lines - 1)) * usable_w)
                for i in range(n_lines)
            ]
        else:
            x_positions = [
            pitch_w - margin_x - int((i / (n_lines - 1)) * usable_w)
                for i in range(n_lines)
            ]

        dot_radius = 16
        font = cv2.FONT_HERSHEY_SIMPLEX

        for line_idx, (n_players, x) in enumerate(zip(all_lines, x_positions)):
            if n_players == 1:
                y_positions = [pitch_h // 2]
            else:
                y_positions = [
                    margin_y + int((j / (n_players - 1)) * usable_h)
                    for j in range(n_players)
                ]

            for y in y_positions:
            # Shadow
                cv2.circle(pitch, (x + 2, y + 2), dot_radius, (0, 60, 0), -1)
                # White ring
                cv2.circle(pitch, (x, y), dot_radius + 3, white, -1)
                # Coloured dot
                cv2.circle(pitch, (x, y), dot_radius, dot_color, -1)
                # Shine effect — small white highlight
                cv2.circle(pitch, (x - 4, y - 4), 4, (255, 255, 255), -1)

    # Team label — top center, clear of dots
        label_size = cv2.getTextSize(team_label, font, 0.85, 2)[0]
        label_x = (pitch_w - label_size[0]) // 2
        cv2.putText(pitch, team_label, (label_x, 48),
                    font, 0.85, white, 2, cv2.LINE_AA)
    
        # Formation string — bottom center, clear of dots
        form_size = cv2.getTextSize(formation_string, font, 0.85, 2)[0]
        form_x = (pitch_w - form_size[0]) // 2
        cv2.putText(pitch, formation_string, (form_x, pitch_h - 22),
                    font, 0.85, white, 2, cv2.LINE_AA)

        return pitch

    def generate_lineup_card(self, save_path='output_videos/lineup_card.png'):
        card_w = 900
        card_h = 620
    
        f1 = self.dominant_formation.get(1, '4-3-3')
        f2 = self.dominant_formation.get(2, '4-3-3')
    
        half1 = self._draw_single_lineup(
            formation_string=f1,
            team_label='Team 1',
            dot_color=(50, 50, 210),
            pitch_w=card_w,
            pitch_h=card_h,
            attacking_direction='right'
        )
    
        half2 = self._draw_single_lineup(
            formation_string=f2,
            team_label='Team 2',
            dot_color=(210, 50, 50),
            pitch_w=card_w,
            pitch_h=card_h,
            attacking_direction='left'
        )
    
        divider = np.zeros((card_h, 8, 3), dtype=np.uint8)
        divider[:] = (220, 220, 220)
    
        combined = np.hstack([half1, divider, half2])
    
        # Dark title bar
        title_bar = np.zeros((70, combined.shape[1], 3), dtype=np.uint8)
        title_bar[:] = (15, 15, 15)
    
        # VS text centred
        title_text = f"{f1}   vs   {f2}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(title_text, font, 1.1, 2)[0]
        text_x = (combined.shape[1] - text_size[0]) // 2
        cv2.putText(title_bar, title_text, (text_x, 48),
                    font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    
        # Team 1 label left
        cv2.putText(title_bar, 'Team 1', (30, 48),
                    font, 0.85, (100, 100, 255), 2, cv2.LINE_AA)
    
        # Team 2 label right
        t2_size = cv2.getTextSize('Team 2', font, 0.85, 2)[0]
        cv2.putText(title_bar, 'Team 2',
                    (combined.shape[1] - t2_size[0] - 30, 48),
                    font, 0.85, (255, 100, 100), 2, cv2.LINE_AA)
    
        final = np.vstack([title_bar, combined])
    
        cv2.imwrite(save_path, final)
        print(f"Lineup card saved: {save_path}")
        return final
    
    def _draw_pitch_markings(self, pitch):
        h, w = pitch.shape[:2]
        color = (255, 255, 255)
        thick = 2

        cv2.rectangle(pitch, (40, 40), (w - 40, h - 40), color, thick)
        cv2.line(pitch, (w // 2, 40), (w // 2, h - 40), color, thick)
        cv2.circle(pitch, (w // 2, h // 2), 80, color, thick)
        cv2.circle(pitch, (w // 2, h // 2), 4, color, -1)

        box_w, box_h = 110, 260
        box_y = (h - box_h) // 2
        cv2.rectangle(pitch, (40, box_y),
                      (40 + box_w, box_y + box_h), color, thick)
        cv2.rectangle(pitch, (w - 40 - box_w, box_y),
                      (w - 40, box_y + box_h), color, thick)

        sy_w, sy_h = 40, 120
        sy_y = (h - sy_h) // 2
        cv2.rectangle(pitch, (40, sy_y),
                      (40 + sy_w, sy_y + sy_h), color, thick)
        cv2.rectangle(pitch, (w - 40 - sy_w, sy_y),
                      (w - 40, sy_y + sy_h), color, thick)

        return pitch

    def draw_formation_overlay_on_video(self, video_frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (420, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            t1 = self.dominant_formation.get(1, 'Unknown')
            t2 = self.dominant_formation.get(2, 'Unknown')

            cv2.putText(frame, f"Team 1: {t1}  |  Team 2: {t2}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

            output_frames.append(frame)

        return output_frames