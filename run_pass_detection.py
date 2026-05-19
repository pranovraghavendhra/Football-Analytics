import pickle
import numpy as np
from pass_detector import PassDetector
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from utils import read_video, get_center_of_bbox, get_foot_position


def main():
    print("Loading stubs...")
    with open('stubs/track_stubs.pkl', 'rb') as f:
        tracks = pickle.load(f)

    print("Stubs loaded. Reading video...")
    video_frames = read_video('input_videos/Football_Dataset.mp4')

    print("Adding positions to tracks...")
    for object, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
            for track_id, track_info in track.items():
                bbox = track_info['bbox']
                if object == 'ball':
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                tracks[object][frame_num][track_id]['position'] = position

    print("Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    print("Applying view transformation...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    print("Assigning teams...")
    team_assigner = TeamAssigner()

    first_player_frame_index = -1
    for frame_num, players_in_frame in enumerate(tracks['players']):
        if len(players_in_frame) > 0:
            first_player_frame_index = frame_num
            break

    if first_player_frame_index == -1:
        print("No players found in stubs.")
        return

    team_assigner.assign_team_color(video_frames[first_player_frame_index],
                                    tracks['players'][first_player_frame_index])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    print("Assigning ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_frame = tracks['ball'][frame_num]

        if 1 not in ball_frame:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox = ball_frame[1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    print("Detecting passes...")
    pass_detector = PassDetector()
    passes = pass_detector.detect_passes(tracks)
    summary = pass_detector.get_pass_summary()

    print("\n--- Pass Detection Summary ---")
    print(f"Total passes:       {summary.get('total_passes', 0)}")
    print(f"Team 1 passes:      {summary.get('team1_passes', 0)}")
    print(f"Team 2 passes:      {summary.get('team2_passes', 0)}")
    print(f"Interceptions:      {summary.get('interceptions', 0)}")
    print(f"Avg pass distance:  {summary.get('avg_pass_distance_meters', 'N/A')} m")
    print(f"Outside polygon:    {summary.get('passes_outside_polygon', 0)} passes (edge of camera view)")
    print("-----------------------------\n")

    print("--- Detailed Pass Log ---")
    for p in summary.get('all_passes', []):
        dist = f"{p['distance_meters']}m" if p['distance_meters'] is not None else "outside pitch polygon"
        print(f"Frame {p['frame']:4d} | {p['type']:12s} | "
              f"Player {p['passer_id']:3d} -> Player {p['receiver_id']:3d} | "
              f"Team {p['passer_team']} -> Team {p['receiver_team']} | "
              f"Dist: {dist}")


if __name__ == '__main__':
    main()