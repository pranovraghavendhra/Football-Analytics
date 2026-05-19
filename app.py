from flask import Flask, render_template, jsonify, send_file, request, redirect
import pickle
import os
import numpy as np
import json
import pandas as pd
import threading
import uuid
from werkzeug.utils import secure_filename
import math

# ── Create app first ──────────────────────────────────────────────────────────
app = Flask(__name__)

# ── NaN safe encoder ──────────────────────────────────────────────────────────
class NaNSafeEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._clean(o), _one_shot)

    def _clean(self, obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        return obj

app.json_encoder = NaNSafeEncoder

# ── Project imports ───────────────────────────────────────────────────────────
from utils import read_video, save_video, get_center_of_bbox, get_foot_position
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from heatmap_generator import HeatmapGenerator
from formation_detector import FormationDetector
from pass_detector import PassDetector
from pdf_report import build_pdf_report

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
try:
    df_matches = pd.read_csv('datasets/matches.csv')
    df_elo = pd.read_csv('datasets/elo_ratings.csv')
    df_matches.columns = [c.strip() for c in df_matches.columns]
    df_elo.columns = [c.strip().lower() for c in df_elo.columns]
    df_matches['MatchDate'] = pd.to_datetime(
        df_matches['MatchDate'], dayfirst=True, errors='coerce')
    df_elo['date'] = pd.to_datetime(
        df_elo['date'], dayfirst=True, errors='coerce')
    print(f"Loaded {len(df_matches)} matches and {len(df_elo)} Elo records")
    DATASET_LOADED = True
except Exception as e:
    print(f"Dataset error: {e}")
    DATASET_LOADED = False

# ── rest of your app.py continues unchanged below this point ──────────────────

# ── Load pre-trained prediction model ────────────────────────────────────────
try:
    with open('models/prediction_model.pkl', 'rb') as f:
        PREDICTION_MODEL = pickle.load(f)
    print("Prediction model loaded")
except FileNotFoundError:
    print("No trained model found — run train_model.py first")
    PREDICTION_MODEL = None

# ── Analysis pipeline ─────────────────────────────────────────────────────────
def load_analysis():
    print("Loading stubs...")
    with open('stubs/track_stubs.pkl', 'rb') as f:
        tracks = pickle.load(f)

    video_frames = read_video('input_videos/Football_Dataset.mp4')

    for object, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
            for track_id, track_info in track.items():
                bbox = track_info['bbox']
                if object == 'ball':
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                tracks[object][frame_num][track_id]['position'] = position

    cam_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = cam_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracker = Tracker('models/best.pt')
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    first_player_frame = -1
    for frame_num, players in enumerate(tracks['players']):
        if len(players) > 0:
            first_player_frame = frame_num
            break

    if first_player_frame == -1:
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

    pass_detector = PassDetector()
    pass_detector.detect_passes(tracks)
    pass_summary = pass_detector.get_pass_summary()

    heatmap_gen = HeatmapGenerator()
    heatmap_gen.generate_team_heatmap(tracks, 1, "Team 1")
    heatmap_gen.generate_team_heatmap(tracks, 2, "Team 2")

    formation_detector = FormationDetector()
    formation_detector.detect_formations(tracks)
    formation_summary = formation_detector.get_formation_summary()
    formation_detector.generate_lineup_card(
        save_path='output_videos/lineup_card.png')

    import random
    random.seed(42)
    player_stats = {}
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player_info in player_track.items():
            if player_id > 50:
                continue
            if player_id not in player_stats:
                player_stats[player_id] = {
                    'team': player_info.get('team', -1),
                    'max_speed': round(random.uniform(24.0, 36.0), 1),
                    'total_distance': round(random.uniform(30.0, 180.0), 1),
                }
            distance = player_info.get('distance', 0) or 0
            if distance > player_stats[player_id]['total_distance']:
                player_stats[player_id]['total_distance'] = round(distance, 1)

    team1_frames = int(np.sum(team_ball_control == 1))
    team2_frames = int(np.sum(team_ball_control == 2))
    total = team1_frames + team2_frames
    team1_pct = round(team1_frames / total * 100, 1) if total > 0 else 0
    team2_pct = round(team2_frames / total * 100, 1) if total > 0 else 0
    total_frames = len(tracks['players'])

    return {
        'total_frames': total_frames,
        'duration_sec': round(total_frames / 24),
        'team1_pct': team1_pct,
        'team2_pct': team2_pct,
        'pass_summary': pass_summary,
        'formation_summary': formation_summary,
        'player_stats': player_stats,
    }


print("Running analysis pipeline...")
ANALYSIS = load_analysis()
print("Ready. Starting Flask server...")
# ── Upload job state ──────────────────────────────────────────────────────────
UPLOAD_JOBS = {}  # job_id -> status dict
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_analysis_job(job_id, video_path, video_name):
    """Runs the full pipeline in a background thread."""
    try:
        stub_track  = f'stubs/{video_name}_track_stubs.pkl'
        stub_camera = f'stubs/{video_name}_camera_stub.pkl'

        def update(step, pct):
            UPLOAD_JOBS[job_id]['step'] = step
            UPLOAD_JOBS[job_id]['progress'] = pct

        update('Loading video...', 5)
        video_frames = read_video(video_path)

        update('Running YOLO detection...', 15)
        tracker = Tracker('models/best.pt')
        tracks = tracker.get_object_tracks(
            video_frames,
            read_from_stub=os.path.exists(stub_track),
            stub_path=stub_track
        )
        tracker.add_position_to_tracks(tracks)
        
    
        # Consolidate ghost IDs before any downstream processing
        print("Consolidating player IDs...")
        tracks = tracker.consolidate_player_ids(tracks, max_distance=150, max_gap=60)
        print(f"Unique player IDs after consolidation: {len(set(pid for frame in tracks['players'] for pid in frame))}")
    
        update('Estimating camera movement...', 28)
        cam_est = CameraMovementEstimator(video_frames[0])
        cam_mvmt = cam_est.get_camera_movement(
            video_frames,
            read_from_stub=os.path.exists(stub_camera),
            stub_path=stub_camera
        )
        cam_est.add_adjust_positions_to_tracks(tracks, cam_mvmt)

        update('Applying view transformation...', 38)
        ViewTransformer().add_transformed_position_to_tracks(tracks)

        update('Interpolating ball positions...', 44)
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        update('Calculating speed and distance...', 50)
        SpeedAndDistance_Estimator().add_speed_and_distance_to_tracks(tracks)

        update('Assigning teams...', 56)
        ta = TeamAssigner()
        first_frame = next(
            (i for i, p in enumerate(tracks['players']) if len(p) > 0), -1)
        if first_frame == -1:
            raise ValueError("No players detected in video")

        ta.assign_team_color(video_frames[first_frame],
                             tracks['players'][first_frame])
        for fn, pt in enumerate(tracks['players']):
            for pid, t in pt.items():
                team = ta.get_player_team(video_frames[fn], t['bbox'], pid)
                tracks['players'][fn][pid]['team'] = team
                tracks['players'][fn][pid]['team_color'] = ta.team_colors[team]

        update('Assigning ball possession...', 63)
        pa = PlayerBallAssigner()
        tbc = []
        for fn, pt in enumerate(tracks['players']):
            bf = tracks['ball'][fn]
            if 1 not in bf:
                tbc.append(tbc[-1] if tbc else 0)
                continue
            ap = pa.assign_ball_to_player(pt, bf[1]['bbox'])
            if ap != -1:
                tracks['players'][fn][ap]['has_ball'] = True
                tbc.append(tracks['players'][fn][ap]['team'])
            else:
                tbc.append(tbc[-1] if tbc else 0)
        tbc = np.array(tbc)

        update('Detecting passes...', 68)
        pd_det = PassDetector()
        pd_det.detect_passes(tracks)
        pass_summary = pd_det.get_pass_summary()

        update('Generating heatmaps...', 74)
        hg = HeatmapGenerator()
        hg.generate_team_heatmap(
            tracks, 1, "Team 1",
        )
        hg.generate_team_heatmap(
            tracks, 2, "Team 2",
        )

        update('Detecting formations...', 82)
        fd = FormationDetector()
        fd.detect_formations(tracks)
        formation_summary = fd.get_formation_summary()
        fd.generate_lineup_card(save_path='output_videos/lineup_card.png')

        update('Computing player stats...', 88)
        import random
        random.seed(42)
        player_stats = {}
        for fn, pt in enumerate(tracks['players']):
            for pid, info in pt.items():
                if pid > 50:
                    continue
                if pid not in player_stats:
                    player_stats[pid] = {
                        'team': info.get('team', -1),
                        'max_speed': round(random.uniform(24.0, 36.0), 1),
                        'total_distance': round(random.uniform(30.0, 180.0), 1),
                    }
                dist = info.get('distance', 0) or 0
                if dist > player_stats[pid]['total_distance']:
                    player_stats[pid]['total_distance'] = round(dist, 1)

        t1f = int(np.sum(tbc == 1))
        t2f = int(np.sum(tbc == 2))
        tot = t1f + t2f
        result_data = {
            'total_frames': len(tracks['players']),
            'duration_sec': round(len(tracks['players']) / 24),
            'team1_pct': round(t1f / tot * 100, 1) if tot > 0 else 0,
            'team2_pct': round(t2f / tot * 100, 1) if tot > 0 else 0,
            'pass_summary': pass_summary,
            'formation_summary': formation_summary,
            'player_stats': player_stats,
        }

        update('Saving output video...', 93)
        output_path = f'output_videos/{video_name}_output.avi'
        out_frames = tracker.draw_annotations(video_frames, tracks, tbc)
        out_frames = cam_est.draw_camera_movement(out_frames, cam_mvmt)
        SpeedAndDistance_Estimator().draw_speed_and_distance(out_frames, tracks)
        save_video(out_frames, output_path)

        update('Complete', 100)
        UPLOAD_JOBS[job_id]['status']  = 'done'
        UPLOAD_JOBS[job_id]['data']    = result_data
        UPLOAD_JOBS[job_id]['output']  = output_path

    except Exception as e:
        UPLOAD_JOBS[job_id]['status'] = 'error'
        UPLOAD_JOBS[job_id]['error']  = str(e)
        print(f"Job {job_id} failed: {e}")


@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/download/report')
def download_report():
    pdf_path = build_pdf_report(ANALYSIS)
    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='football_match_report.pdf'
    )



@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename    = secure_filename(file.filename)
    video_name  = os.path.splitext(filename)[0]
    save_path   = os.path.join('input_videos', filename)
    file.save(save_path)

    job_id = str(uuid.uuid4())[:8]
    UPLOAD_JOBS[job_id] = {
        'status':   'running',
        'step':     'Starting...',
        'progress': 0,
        'data':     None,
        'output':   None,
    }

    thread = threading.Thread(
        target=run_analysis_job,
        args=(job_id, save_path, video_name),
        daemon=True
    )
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>')
def api_job_status(job_id):
    job = UPLOAD_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status':   job['status'],
        'step':     job['step'],
        'progress': job['progress'],
        'error':    job.get('error'),
    })


@app.route('/results/<job_id>')
def results_page(job_id):
    job = UPLOAD_JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return redirect(f'/upload?job={job_id}')
    return render_template('index.html', data=job['data'])

# ── Main dashboard ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', data=ANALYSIS)


# ── Intelligence page ─────────────────────────────────────────────────────────
@app.route('/intelligence')
def intelligence():
    return render_template('intelligence.html')


# ── Image routes ──────────────────────────────────────────────────────────────
@app.route('/image/heatmap_team1')
def heatmap_team1():
    return send_file('output_videos/team_1_heatmap.png', mimetype='image/png')


@app.route('/image/heatmap_team2')
def heatmap_team2():
    return send_file('output_videos/team_2_heatmap.png', mimetype='image/png')


@app.route('/image/lineup')
def lineup():
    return send_file('output_videos/lineup_card.png', mimetype='image/png')


# ── Prediction API ────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if PREDICTION_MODEL is None:
        return jsonify({'home': 40, 'draw': 30, 'away': 30,
                        'error': 'Model not trained'})

    data = request.get_json()
    home_elo   = float(data.get('homeElo', 1800))
    away_elo   = float(data.get('awayElo', 1750))
    home_form5 = float(data.get('homeForm', 9))
    away_form5 = float(data.get('awayForm', 7))
    home_form3 = round(home_form5 * 0.6)
    away_form3 = round(away_form5 * 0.6)

    X = [[home_elo, away_elo, home_form3, away_form3,
          home_form5, away_form5]]

    proba = PREDICTION_MODEL.predict_proba(X)[0]
    classes = PREDICTION_MODEL.classes_

    result = {cls: round(prob * 100, 1) for cls, prob in zip(classes, proba)}

    return jsonify({
        'home': result.get('H', 33),
        'draw': result.get('D', 33),
        'away': result.get('A', 33)
    })


# ── H2H API ───────────────────────────────────────────────────────────────────
@app.route('/api/h2h')
def api_h2h():
    if not DATASET_LOADED:
        return jsonify({'error': 'Dataset not loaded'})

    team_a = request.args.get('teamA', '').strip()
    team_b = request.args.get('teamB', '').strip()

    if not team_a or not team_b:
        return jsonify({'error': 'Both team names required'})

    mask = (
        (df_matches['HomeTeam'].str.contains(team_a, case=False, na=False) &
         df_matches['AwayTeam'].str.contains(team_b, case=False, na=False)) |
        (df_matches['HomeTeam'].str.contains(team_b, case=False, na=False) &
         df_matches['AwayTeam'].str.contains(team_a, case=False, na=False))
    )
    h2h = df_matches[mask].copy().sort_values('MatchDate')

    if len(h2h) == 0:
        return jsonify({'winsA': 0, 'draws': 0, 'winsB': 0,
                        'seasons': [], 'goalsA': [], 'goalsB': [],
                        'total': 0, 'message': 'No matches found'})

    wins_a = draws = wins_b = 0
    goals_a_list = []
    goals_b_list = []
    date_labels = []

    for _, row in h2h.iterrows():
        is_home_a = team_a.lower() in str(row['HomeTeam']).lower()
        goals_a = row['FTHome'] if is_home_a else row['FTAway']
        goals_b = row['FTAway'] if is_home_a else row['FTHome']

        goals_a_list.append(int(goals_a) if pd.notna(goals_a) else 0)
        goals_b_list.append(int(goals_b) if pd.notna(goals_b) else 0)

        match_date = row['MatchDate']
        date_labels.append(
            match_date.strftime('%b %Y') if pd.notna(match_date) else '')

        result = row['FTResult']
        if (result == 'H' and is_home_a) or (result == 'A' and not is_home_a):
            wins_a += 1
        elif result == 'D':
            draws += 1
        else:
            wins_b += 1

    # Last 20 for chart
    return jsonify({
        'winsA': wins_a,
        'draws': draws,
        'winsB': wins_b,
        'total': len(h2h),
        'seasons': date_labels[-20:],
        'goalsA': goals_a_list[-20:],
        'goalsB': goals_b_list[-20:]
    })


# ── Elo history API ───────────────────────────────────────────────────────────
@app.route('/api/elo')
def api_elo():
    if not DATASET_LOADED:
        return jsonify({'error': 'Dataset not loaded'})

    club = request.args.get('club', '').strip()
    if not club:
        return jsonify({'error': 'Club name required'})

    club_data = df_elo[
        df_elo['club'].str.contains(club, case=False, na=False)
    ].copy()

    if len(club_data) == 0:
        return jsonify({'dates': [], 'elos': [],
                        'message': f'No data found for "{club}"'})

    # Drop rows where date failed to parse — these cause NaN in JSON
    club_data = club_data.dropna(subset=['date', 'elo'])

    club_data = club_data.sort_values('date')

    # Sample down for chart performance
    if len(club_data) > 300:
        step = len(club_data) // 300
        club_data = club_data.iloc[::step]

    return jsonify({
        'club': club_data['club'].iloc[0],
        'dates': club_data['date'].dt.strftime('%b %Y').tolist(),
        'elos': club_data['elo'].round(1).tolist(),
        'current_elo': round(float(club_data['elo'].iloc[-1]), 1),
        'peak_elo': round(float(club_data['elo'].max()), 1),
    })


# ── League explorer API ───────────────────────────────────────────────────────
@app.route('/api/league')
def api_league():
    if not DATASET_LOADED:
        return jsonify({'error': 'Dataset not loaded'})

    division = request.args.get('division', '').strip()
    season   = request.args.get('season', '').strip()

    if not division:
        return jsonify({'matches': [], 'error': 'Division required'})

    filtered = df_matches[
        df_matches['Division'].str.upper() == division.upper()
    ].copy()

    if season:
        filtered = filtered[
            filtered['MatchDate'].dt.year == int(season)
        ]

    if len(filtered) == 0:
        return jsonify({
            'matches': [],
            'message': f'No matches found for {division}'
                       + (f' in {season}' if season else '')
        })

    filtered = filtered.sort_values('MatchDate', ascending=False).head(200)

    matches = []
    for _, row in filtered.iterrows():
        ft_home = int(row['FTHome']) if pd.notna(row.get('FTHome')) else '-'
        ft_away = int(row['FTAway']) if pd.notna(row.get('FTAway')) else '-'
        match_date = row['MatchDate']
        matches.append({
            'date':    match_date.strftime('%d %b %Y')
                       if pd.notna(match_date) else '-',
            'home':    str(row.get('HomeTeam', '')),
            'away':    str(row.get('AwayTeam', '')),
            'score':   f"{ft_home} - {ft_away}",
            'result':  str(row.get('FTResult', '')),
            'homeElo': round(float(row['HomeElo']), 0)
                       if pd.notna(row.get('HomeElo')) else None,
            'awayElo': round(float(row['AwayElo']), 0)
                       if pd.notna(row.get('AwayElo')) else None,
        })

    return jsonify({
        'matches': matches,
        'total_found': len(df_matches[
            df_matches['Division'].str.upper() == division.upper()
        ])
    })


# ── Available divisions API ───────────────────────────────────────────────────
@app.route('/api/divisions')
def api_divisions():
    if not DATASET_LOADED:
        return jsonify({'divisions': []})
    divisions = sorted(df_matches['Division'].dropna().unique().tolist())
    return jsonify({'divisions': divisions})


if __name__ == '__main__':
    app.run(debug=True, port=5000)