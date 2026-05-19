import pickle
import cv2
import numpy as np
from formation_detector import FormationDetector
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from utils import read_video, get_center_of_bbox, get_foot_position


def main():
    print("Loading stubs...")
    with open('stubs/track_stubs.pkl', 'rb') as f:
        tracks = pickle.load(f)

    print("Reading video...")
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
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame)

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
        print("No players found.")
        return

    team_assigner.assign_team_color(
        video_frames[first_player_frame_index],
        tracks['players'][first_player_frame_index]
    )

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team

    print("Detecting formations...")
    formation_detector = FormationDetector()
    formation_detector.detect_formations(tracks)
    summary = formation_detector.get_formation_summary()

    print("\n--- Formation Detection Summary ---")
    for team_id in [1, 2]:
        team_summary = summary[team_id]
        print(f"\nTeam {team_id}:")
        print(f"  Dominant formation : {team_summary['dominant_formation']}")
        print(f"  Total samples      : {team_summary['total_samples']}")
        print(f"  Formation breakdown:")
        for formation, count in team_summary['formation_counts'].items():
            pct = round(count / team_summary['total_samples'] * 100, 1)
            print(f"    {formation:12s} — {count:3d} frames ({pct}%)")

    print("\nGenerating lineup card...")
    formation_detector.generate_lineup_card(
        save_path='output_videos/lineup_card.png'
    )

    print("\nDone! Check output_videos folder for lineup_card.png")


if __name__ == '__main__':
    main()