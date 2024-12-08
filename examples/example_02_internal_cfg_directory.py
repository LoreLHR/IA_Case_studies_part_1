#################################################################################
# Example of using the internal config directory of pyppbox package
#################################################################################

import cv2

from pyppbox.standalone import setConfigDir, detectPeople, trackPeople, reidPeople
from pyppbox.utils.visualizetools import visualizePeople


# The config directory is a directory where consists of 4 required YAML files:
#   - main.yaml, indicates which detector/tracker/reider is used.
#   - detectors.yaml, stores all detectors' configurations.
#   - trackers.yaml, stores all trackers' configurations.
#   - reiders.yaml, stores all reiders' configurations.
#
# The internal config directory of pyppbox is '{pyppbox root}/config/cfg'.

# Use the internal config directory of pyppbox
setConfigDir(config_dir=None, load_all=True)

input_video=r"C:\Users\Loren\Downloads\wetransfer_img_2100-mov_2024-11-18_1550\IMG_2102.mov"
cap = cv2.VideoCapture(0)

frame_id=0

while cap.isOpened():
    frame_id=frame_id+1
    hasFrame, frame = cap.read()

    if hasFrame:

        # Detect people without visualizing
        detected_people, _ = detectPeople(frame, img_is_mat=True, visual=False)

        # Track the detected people
        tracked_people = trackPeople(frame, detected_people, img_is_mat=True)

        # Re-identify the tracked people
        reidentified_people, reid_count = reidPeople(
            frame, 
            tracked_people, 
            img_is_mat=True
        ) 



        i=0
        for p in tracked_people:
            i=i+1
            print('person_------------------------')
            print("box 1 ="+str(p.box_xyxy))
            print("box 2 ="+str(p.box_xywh))
            print("box 3 :"+str(p.faceid))
            print("box 4 :"+str(p.deepid))

            cropped_image = frame[p.box_xyxy[1]:p.box_xyxy[3],p.box_xyxy[0]:p.box_xyxy[2]]
            img_name="data/lolov2/"+str(frame_id)+"_"+str(i)+ ".jpg"
            cv2.imwrite(img_name, cropped_image)
                
            print(img_name)

        # Visualize people in video frame with reid status `show_reid=reid_count`
        visualized_mat = visualizePeople(
            frame, 
            reidentified_people, 
            show_reid=reid_count
        )
        cv2.imshow("pyppbox: example_02_internal_cfg_directory.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

