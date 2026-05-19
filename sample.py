import cv2

cap = cv2.VideoCapture('input_videos/Football_Dataset.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cv2.imwrite('output_videos/sample_frame.png', frame)
print(f"Resolution: {frame.shape[1]} x {frame.shape[0]}")
cap.release()
print("Saved sample_frame.png")