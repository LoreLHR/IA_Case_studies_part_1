import cv2
import os
import time
import cv2
from pyppbox.standalone import setConfigDir, detectPeople, trackPeople, reidPeople
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.standalone import trainReIDClassifier

setConfigDir(config_dir="cfg", load_all=True)

import os
import cv2

myreider = {
    'ri_name': 'Torchreid',
    'classifier_pkl': r'C:\Users\Loren\anaconda3\envs\pyppbox\Lib\site-packages\pyppbox\data\modules\torchreid\classifier\test4.pkl',
    'train_data': r'C:/Users\Loren\anaconda3\envs\pyppbox\Lib\site-packages\pyppbox\data\datasets\TEST_GROUP\body_128x256',
    'model_name': 'osnet_ain_x1_0',
    'model_path': r'C:\Users\Loren\anaconda3\envs\pyppbox\Lib\site-packages\pyppbox\data\modules\torchreid\models\torchreid\osnet_ain_ms_d_c.pth.tar',
    'min_confidence': 0.70,
    'device': 'cuda'
}

def set_webcam_resolution(cap, width, height):
    # Set the width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def resize(folder, size=(128, 256)):
    # Check if the folder exists
    if not os.path.exists(folder):
        print(f"The folder {folder} does not exist.")
        return

    # Loop through the files in the folder
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        
        # Check if it is an image file
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            # Resize the image
            resized_img = cv2.resize(img, size)

            # Overwrite the original image
            cv2.imwrite(img_path, resized_img)
            print(f"Image resized and saved: {img_path}")

def main():
    print("Launching Pyppbox...")
    # Start the camera
    cap = cv2.VideoCapture(1)
    set_webcam_resolution(cap, 1080, 1080)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return
    capture = False
    frame_id = 0
    i = 0
    print("Do you want to register yourself? (press o): ")
    while cap.isOpened():
        hasFrame, frame = cap.read()
        clean_frame = frame.copy()
        frame_id = frame_id + 1
        if hasFrame:
            # Step 1: Detect people
            detected_people, _ = detectPeople(frame, img_is_mat=True, visual=False)

            # Step 2: Track detected people
            tracked_people = trackPeople(frame, detected_people, img_is_mat=True)

            # Step 3: Re-identify tracked people
            reidentified_people, reid_count = reidPeople(
                frame,
                tracked_people,
                img_is_mat=True
            ) 
            # Step 4: Visualize people
            visualized_mat = visualizePeople(
                frame,
                reidentified_people,
                show_ids=(False, True, False),
                show_reid=reid_count
            )
            # Show the video with annotations
            cv2.imshow("Pyppbox - Live", visualized_mat)

            if cv2.waitKey(1) == ord('o'):
                capture = True
                i = 0
                name = input("Enter your name: ")
                folder = "person_data/" + name
                os.makedirs(folder, exist_ok=True)
                print("Please step back to be fully visible.")
                time.sleep(2)
                # Train the new model
                print("Get ready! Image capture will start in 10 seconds. Make sure no one else is in front of the camera. Turn, move, have fun!")
                time.sleep(3)  # Time for preparation
                print("Capturing images...")
            start_time = time.time()
            images = []
            
            if capture == True:
                i = i + 1
                for p in tracked_people:
                    if i % 4 == 0:
                        cropped_image = clean_frame[p.box_xyxy[1]:p.box_xyxy[3], p.box_xyxy[0]:p.box_xyxy[2]]
                        
                        img_name = "person_data/" + name + '/' + str(frame_id) + "_" + str(i) + ".jpg"
                        cv2.imwrite(img_name, cropped_image)
                        print(img_name)
            if i == 400:
                capture = False
                print("Capture completed.")

                print("Resizing images...")
                resize(folder)

                print("Training the model...")
                model_name = "model_reid_made_by_lolo.pkl"
                trainReIDClassifier(
                    reider=myreider, 
                    train_data="person_data", # Set train_data="" means using the default 'train_data' in line 12
                    classifier_pkl='classifier/' + model_name # Set classifier_pkl="" to use the default in line 14
                )
                print("Model trained and saved.")
                i = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

if __name__ == "__main__":
    main()
