from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=90,        # was default 30 — keeps lost tracks for 3.75s at 24fps
            minimum_matching_threshold=0.8,
            frame_rate=24
        )

    def consolidate_player_ids(self, tracks, max_distance=150, max_gap=60):
        """
        Merges ghost IDs back into their original player IDs.
        
        Logic: if a high ID player appears near where a low ID player
        disappeared, and the gap between disappearance and reappearance
        is small, they are likely the same player.
        """
        player_tracks = tracks['players']
        total_frames  = len(player_tracks)
    
        # Build last-seen record for every player ID
        last_seen = {}   # player_id -> {'frame': N, 'bbox': [...]}
        first_seen = {}  # player_id -> frame number
    
        for frame_num, frame in enumerate(player_tracks):
            for pid, info in frame.items():
                if pid not in first_seen:
                    first_seen[pid] = frame_num
                last_seen[pid] = {
                    'frame': frame_num,
                    'bbox':  info['bbox']
                }
    
        # Separate low IDs (likely real original tracks)
        # from high IDs (likely ghost reassignments)
        # Threshold: any ID above 30 that appears after frame 10
        # is a candidate ghost
        ghost_ids = {
            pid for pid in first_seen
            if pid > 30 and first_seen[pid] > 10
        }
        real_ids  = {pid for pid in first_seen if pid not in ghost_ids}
    
        # For each ghost, find the best matching real ID
        id_map = {}  # ghost_id -> real_id
    
        for ghost_id in ghost_ids:
            ghost_first_frame = first_seen[ghost_id]
            ghost_first_bbox  = None
            for frame_num in range(ghost_first_frame,
                                   min(ghost_first_frame + 5, total_frames)):
                if ghost_id in player_tracks[frame_num]:
                    ghost_first_bbox = player_tracks[frame_num][ghost_id]['bbox']
                    break
    
            if ghost_first_bbox is None:
                continue
    
            ghost_cx = (ghost_first_bbox[0] + ghost_first_bbox[2]) / 2
            ghost_cy = (ghost_first_bbox[1] + ghost_first_bbox[3]) / 2
    
            best_match = None
            best_score = float('inf')
    
            for real_id in real_ids:
                # Real ID must have disappeared before ghost appeared
                real_last = last_seen[real_id]
                gap = ghost_first_frame - real_last['frame']
    
                # Gap must be within allowed range
                if gap < 0 or gap > max_gap:
                    continue
    
                # Calculate distance between where real disappeared
                # and where ghost appeared
                real_bbox = real_last['bbox']
                real_cx = (real_bbox[0] + real_bbox[2]) / 2
                real_cy = (real_bbox[1] + real_bbox[3]) / 2
    
                dist = ((ghost_cx - real_cx)**2 +
                        (ghost_cy - real_cy)**2) ** 0.5
    
                if dist < max_distance and dist < best_score:
                    best_score = dist
                    best_match = real_id
    
            if best_match is not None:
                id_map[ghost_id] = best_match
    
        # Apply the ID map — replace ghost IDs with real IDs in all frames
        if not id_map:
            print("No ghost IDs found to consolidate")
            return tracks
    
        print(f"Consolidating {len(id_map)} ghost IDs: {id_map}")
    
        for frame_num, frame in enumerate(player_tracks):
            # Collect replacements to avoid dict mutation during iteration
            to_add    = {}
            to_remove = []
    
            for pid, info in frame.items():
                if pid in id_map:
                    real_id = id_map[pid]
                    # Only replace if real ID not already in this frame
                    if real_id not in frame:
                        to_add[real_id] = info
                    to_remove.append(pid)
    
            for pid in to_remove:
                del tracks['players'][frame_num][pid]
            for pid, info in to_add.items():
                tracks['players'][frame_num][pid] = info
    
        return tracks
    
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total if total > 0 else 0
        team_2 = team_2_num_frames / total if total > 0 else 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames