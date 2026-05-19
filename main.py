from utils import read_video, save_video, measure_distance, get_center_of_bbox
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def filter_ball_outliers(tracks, max_pixel_jump=120):
    last_valid_center = None
    last_valid_frame = None

    for frame_num, ball_frame in enumerate(tracks['ball']):
        if 1 not in ball_frame:
            continue

        bbox = ball_frame[1]['bbox']
        center = get_center_of_bbox(bbox)

        if last_valid_center is not None:
            dist = measure_distance(center, last_valid_center)
            frames_elapsed = frame_num - last_valid_frame
            per_frame_jump = dist / max(frames_elapsed, 1)

            if per_frame_jump > max_pixel_jump:
                tracks['ball'][frame_num] = {}
                continue

        last_valid_center = center
        last_valid_frame = frame_num

    return tracks


def main():
    # Read video
    video_frames = read_video('input_videos/Football_Dataset.mp4')
    print(f"Total frames loaded: {len(video_frames)}")

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)

    tracker.add_position_to_tracks(tracks)

# Consolidate ghost IDs before any downstream processing
    print("Consolidating player IDs...")
    tracks = tracker.consolidate_player_ids(tracks, max_distance=150, max_gap=60)
    print(f"Unique player IDs after consolidation: {len(set(pid for frame in tracks['players'] for pid in frame))}")

    # Camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Filter ball outliers then interpolate
    tracks = filter_ball_outliers(tracks, max_pixel_jump=120)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign teams
    team_assigner = TeamAssigner()

    first_player_frame_index = -1
    for frame_num, players_in_frame in enumerate(tracks['players']):
        if len(players_in_frame) > 0:
            first_player_frame_index = frame_num
            break

    if first_player_frame_index != -1:
        team_assigner.assign_team_color(
            video_frames[first_player_frame_index],
            tracks['players'][first_player_frame_index])
    else:
        print("No players found in any frame. Check your detection model.")
        return

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = \
                team_assigner.team_colors[team]

    # Ball possession — guarded against empty frames
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_frame = tracks['ball'][frame_num]

        if 1 not in ball_frame:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox = ball_frame[1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # Draw output
    print("Drawing annotations...")
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)

    print("Drawing camera movement...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    print("Drawing speed and distance...")
    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    print("Saving video...")
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print("Done! Output saved to output_videos/output_video.avi")


if __name__ == '__main__':
    main()