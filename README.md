# Group: 

Lorenzo Lhoir, Maxime Cayphas, Mathias Vertenoeuil, Bastien Francotte

# Pyppbox-Based Person Re-Identification System

This project is part of our Master's degree in Computer Vision. It demonstrates a person re-identification pipeline using **Pyppbox**, a framework designed for seamless integration of detection, tracking, and re-identification models. Our goal is to enable the registration and re-identification of individuals through real-time image capture and model training.

---

## **Project Overview**
The system:
1. Detects people in a video stream using Pyppbox.
2. Prompts unknown individuals to register via the terminal.
3. Captures images of the person for training.
4. Trains a new re-identification model with the new individual.
5. Updates and applies the new model dynamically during execution.

---

## **Key Features**
- **Real-Time Detection and Tracking:** Detects and tracks individuals in live video.
- **User Registration:** Prompts unknown users to register their identity.
- **Dynamic Model Training:** Automatically retrains the person re-identification model when a new individual is added.
- **Seamless Re-Identification:** Identifies registered individuals using the updated model.

---

## **Project Structure**
```plaintext
├── examples/
│   ├── main.py          # The main script for detection, tracking, and re-identification.
├── cfg/
│   ├── reid_model.pkl   # The latest re-identification model used by the system.
├── person_data/
│   ├── [User_Folders]   # Folders containing training images of each registered person.
├── classifier/
│   ├── model_reid_made_by_lolo.pkl  # The newly trained re-identification models.
