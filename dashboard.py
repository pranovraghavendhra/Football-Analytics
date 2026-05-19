import streamlit as st
import pickle
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

from utils import read_video, save_video, measure_distance, get_center_of_bbox, get_foot_position
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from heatmap_generator import HeatmapGenerator
from formation_detector import FormationDetector


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analysis Dashboard",
    page_icon="⚽",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #16213e;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        padding: 12px 0 8px 0;
        border-bottom: 2px solid #00d4aa;
        margin-bottom: 20px;
    }
    .stProgress > div > div {
        background-color: #00d4aa;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper — run full pipeline ────────────────────────────────────────────────
def run_pipeline(video_path, track_stub, camera_stub, progress_bar, status_text):

    status_text.text("Loading video...")
    progress_bar.progress(5)
    video_frames = read_video(video_path)
    total_frames = len(video_frames)

    status_text.text("Running YOLO detection + tracking...")
    progress_bar.progress(15)
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=os.path.exists(track_stub),
        stub_path=track_stub
    )
    tracker.add_position_to_tracks(tracks)

    status_text.text("Estimating camera movement...")
    progress_bar.progress(30)
    cam_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = cam_estimator.get_camera_movement(
        video_frames,
        read_from_stub=os.path.exists(camera_stub),
        stub_path=camera_stub
    )
    cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    status_text.text("Applying view transformation...")
    progress_bar.progress(40)
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    status_text.text("Interpolating ball positions...")
    progress_bar.progress(45)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    status_text.text("Calculating speed and distance...")
    progress_bar.progress(50)
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    status_text.text("Assigning teams...")
    progress_bar.progress(55)
    team_assigner = TeamAssigner()

    first_player_frame = -1
    for frame_num, players in enumerate(tracks['players']):
        if len(players) > 0:
            first_player_frame = frame_num
            break

    if first_player_frame == -1:
        st.error("No players found in video.")
        return None

    team_assigner.assign_team_color(
        video_frames[first_player_frame],
        tracks['players'][first_player_frame]
    )

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = \
                team_assigner.team_colors[team]

    status_text.text("Assigning ball possession...")
    progress_bar.progress(60)
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

    team_ball_control_arr = np.array(team_ball_control)

    status_text.text("Generating heatmaps...")
    progress_bar.progress(70)
    heatmap_gen = HeatmapGenerator()
    heatmap_team1 = heatmap_gen.generate_team_heatmap(
        tracks, team_id=1, team_label="Team 1")
    heatmap_team2 = heatmap_gen.generate_team_heatmap(
        tracks, team_id=2, team_label="Team 2")
    heatmap_gen.generate_combined_heatmap(tracks)

    status_text.text("Detecting formations...")
    progress_bar.progress(80)
    formation_detector = FormationDetector()
    formation_detector.detect_formations(tracks)
    formation_summary = formation_detector.get_formation_summary()
    lineup_card = formation_detector.generate_lineup_card(
        save_path='output_videos/lineup_card.png')

    status_text.text("Computing player stats...")
    progress_bar.progress(88)
    player_stats = compute_player_stats(tracks)

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    return {
        'tracks': tracks,
        'team_ball_control': team_ball_control_arr,
        'heatmap_team1': heatmap_team1,
        'heatmap_team2': heatmap_team2,
        'formation_summary': formation_summary,
        'lineup_card': lineup_card,
        'player_stats': player_stats,
        'total_frames': total_frames,
        'video_frames': video_frames,
        'cam_estimator': cam_estimator,
        'camera_movement_per_frame': camera_movement_per_frame,
        'speed_estimator': speed_estimator,
        'tracker': tracker,
    }


def compute_player_stats(tracks):
    """Extracts max speed and total distance per player."""
    player_stats = {}

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player_info in player_track.items():
            if player_id > 50:
                continue

            if player_id not in player_stats:
                player_stats[player_id] = {
                    'team': player_info.get('team', -1),
                    'max_speed': 0,
                    'total_distance': 0,
                }

            speed = player_info.get('speed', 0)
            distance = player_info.get('distance', 0)

            if speed and speed > player_stats[player_id]['max_speed']:
                player_stats[player_id]['max_speed'] = round(speed, 1)

            if distance and distance > player_stats[player_id]['total_distance']:
                player_stats[player_id]['total_distance'] = round(distance, 1)

    return player_stats


def draw_possession_bar(team1_pct, team2_pct):
    """Draws a simple HTML possession bar."""
    t1 = round(team1_pct * 100, 1)
    t2 = round(team2_pct * 100, 1)
    st.markdown(f"""
    <div style="margin: 10px 0">
        <div style="display:flex; justify-content:space-between;
                    font-size:13px; color:#aaa; margin-bottom:4px">
            <span>Team 1 — {t1}%</span>
            <span>Team 2 — {t2}%</span>
        </div>
        <div style="display:flex; height:28px; border-radius:14px;
                    overflow:hidden; background:#222">
            <div style="width:{t1}%; background:#3b82f6;
                        display:flex; align-items:center;
                        justify-content:center; font-size:12px;
                        font-weight:600; color:white">{t1}%</div>
            <div style="width:{t2}%; background:#ef4444;
                        display:flex; align-items:center;
                        justify-content:center; font-size:12px;
                        font-weight:600; color:white">{t2}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main dashboard ────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("## ⚽ Football Match Analysis Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload a football video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video to analyse"
        )

        use_existing = st.checkbox(
            "Use existing processed video",
            value=True,
            help="Load from stubs if available — much faster"
        )

        if use_existing:
            existing_video = st.selectbox(
                "Select video",
                options=[f for f in os.listdir('input_videos')
                         if f.endswith(('.mp4', '.avi', '.mov'))]
                if os.path.exists('input_videos') else []
            )

        st.markdown("---")
        st.markdown("### 📁 Project")
        st.markdown(f"**Stubs:** {'✅ Found' if os.path.exists('stubs/track_stubs.pkl') else '❌ Not found'}")
        st.markdown(f"**Output video:** {'✅ Found' if os.path.exists('output_videos/output_video.avi') else '❌ Not found'}")
        st.markdown(f"**Lineup card:** {'✅ Found' if os.path.exists('output_videos/lineup_card.png') else '❌ Not found'}")
        st.markdown(f"**Heatmaps:** {'✅ Found' if os.path.exists('output_videos/combined_heatmap.png') else '❌ Not found'}")

        run_button = st.button("▶ Run Analysis", type="primary",
                               use_container_width=True)

    # Main area — show results if analysis was run
    if run_button:
        if uploaded_file:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
                f.write(uploaded_file.read())
                video_path = f.name
            track_stub = 'stubs/uploaded_track_stubs.pkl'
            camera_stub = 'stubs/uploaded_camera_stub.pkl'
        elif use_existing and existing_video:
            video_path = f'input_videos/{existing_video}'
            name = os.path.splitext(existing_video)[0]
            track_stub = f'stubs/{name}_track_stubs.pkl'
            camera_stub = f'stubs/{name}_camera_stub.pkl'
            # Fallback to default stubs
            if not os.path.exists(track_stub):
                track_stub = 'stubs/track_stubs.pkl'
            if not os.path.exists(camera_stub):
                camera_stub = 'stubs/camera_movement_stub.pkl'
        else:
            st.warning("Please upload a video or select an existing one.")
            return

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = run_pipeline(
            video_path, track_stub, camera_stub,
            progress_bar, status_text
        )

        if results:
            st.session_state['results'] = results
            st.session_state['analysed'] = True

    # Display results
    if st.session_state.get('analysed') and st.session_state.get('results'):
        results = st.session_state['results']
        display_results(results)


def display_results(results):
    tracks = results['tracks']
    team_ball_control = results['team_ball_control']
    formation_summary = results['formation_summary']
    player_stats = results['player_stats']
    total_frames = results['total_frames']

    # ── Top metrics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Match Overview</div>',
                unsafe_allow_html=True)

    team1_frames = np.sum(team_ball_control == 1)
    team2_frames = np.sum(team_ball_control == 2)
    total = team1_frames + team2_frames
    team1_pct = team1_frames / total if total > 0 else 0
    team2_pct = team2_frames / total if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Frames", f"{total_frames:,}")
    with col2:
        st.metric("Duration (approx)",
                  f"{round(total_frames / 24)}s")
    with col3:
        st.metric("Team 1 Possession", f"{round(team1_pct * 100, 1)}%")
    with col4:
        st.metric("Team 2 Possession", f"{round(team2_pct * 100, 1)}%")

    # ── Possession bar ───────────────────────────────────────────────────────
    st.markdown("#### Ball Possession")
    draw_possession_bar(team1_pct, team2_pct)

    st.markdown("---")

    # ── Formations ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Formation Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        t1_summary = formation_summary.get(1, {})
        st.markdown("**Team 1**")
        st.metric("Dominant Formation",
                  t1_summary.get('dominant_formation', 'Unknown'))
        if t1_summary.get('formation_counts'):
            for formation, count in list(
                    t1_summary['formation_counts'].items())[:4]:
                pct = round(count / t1_summary['total_samples'] * 100, 1)
                st.progress(pct / 100, text=f"{formation} — {pct}%")

    with col2:
        t2_summary = formation_summary.get(2, {})
        st.markdown("**Team 2**")
        st.metric("Dominant Formation",
                  t2_summary.get('dominant_formation', 'Unknown'))
        if t2_summary.get('formation_counts'):
            for formation, count in list(
                    t2_summary['formation_counts'].items())[:4]:
                pct = round(count / t2_summary['total_samples'] * 100, 1)
                st.progress(pct / 100, text=f"{formation} — {pct}%")

    # Lineup card
    if os.path.exists('output_videos/lineup_card.png'):
        st.markdown("#### Lineup Card")
        lineup_img = Image.open('output_videos/lineup_card.png')
        st.image(lineup_img, use_container_width=True)

    st.markdown("---")

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player Position Heatmaps</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('output_videos/team_1_heatmap.png'):
            st.markdown("**Team 1**")
            img = Image.open('output_videos/team_1_heatmap.png')
            st.image(img, use_container_width=True)

    with col2:
        if os.path.exists('output_videos/team_2_heatmap.png'):
            st.markdown("**Team 2**")
            img = Image.open('output_videos/team_2_heatmap.png')
            st.image(img, use_container_width=True)

    if os.path.exists('output_videos/combined_heatmap.png'):
        st.markdown("**Combined**")
        img = Image.open('output_videos/combined_heatmap.png')
        st.image(img, use_container_width=True)

    st.markdown("---")

    # ── Player stats ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player Speed and Distance</div>',
                unsafe_allow_html=True)

    if player_stats:
        team1_players = {pid: s for pid, s in player_stats.items()
                         if s['team'] == 1}
        team2_players = {pid: s for pid, s in player_stats.items()
                         if s['team'] == 2}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Team 1 Players**")
            if team1_players:
                # Sort by distance
                sorted_players = sorted(
                    team1_players.items(),
                    key=lambda x: x[1]['total_distance'],
                    reverse=True
                )
                for player_id, stats in sorted_players:
                    with st.expander(f"Player {player_id}"):
                        c1, c2 = st.columns(2)
                        c1.metric("Max Speed",
                                  f"{stats['max_speed']} km/h")
                        c2.metric("Distance",
                                  f"{stats['total_distance']} m")
            else:
                st.info("No Team 1 player data available")

        with col2:
            st.markdown("**Team 2 Players**")
            if team2_players:
                sorted_players = sorted(
                    team2_players.items(),
                    key=lambda x: x[1]['total_distance'],
                    reverse=True
                )
                for player_id, stats in sorted_players:
                    with st.expander(f"Player {player_id}"):
                        c1, c2 = st.columns(2)
                        c1.metric("Max Speed",
                                  f"{stats['max_speed']} km/h")
                        c2.metric("Distance",
                                  f"{stats['total_distance']} m")
            else:
                st.info("No Team 2 player data available")

        # Top 3 fastest players
        st.markdown("#### Top 3 Fastest Players")
        all_sorted = sorted(
            player_stats.items(),
            key=lambda x: x[1]['max_speed'],
            reverse=True
        )[:3]

        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, (player_id, stats) in enumerate(all_sorted):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem">{medals[i]}</div>
                    <div class="metric-value">{stats['max_speed']}</div>
                    <div class="metric-label">km/h — Player {player_id}
                    (Team {stats['team']})</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Output video download ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Output Video</div>',
                unsafe_allow_html=True)

    if os.path.exists('output_videos/output_video.avi'):
        with open('output_videos/output_video.avi', 'rb') as f:
            st.download_button(
                label="⬇ Download Output Video",
                data=f,
                file_name="football_analysis_output.avi",
                mime="video/x-msvideo",
                use_container_width=True
            )
    else:
        st.info("Run the analysis first to generate the output video.")


if __name__ == '__main__':
    main()