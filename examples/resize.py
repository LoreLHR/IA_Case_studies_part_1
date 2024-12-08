import cv2
import os

# Liste des dossiers
folders = [ "data/bastienv2"]

# Dimensions cibles
target_size = (128, 256)  # largeur x hauteur

for folder in folders:
    # Vérifie si le dossier existe
    if not os.path.exists(folder):
        print(f"Le dossier {folder} n'existe pas.")
        continue

    # Créer un dossier pour les images redimensionnées (optionnel)
    resized_folder = folder + "_resized"
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    # Parcourt les fichiers du dossier
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        
        # Vérifie si c'est bien un fichier image
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Charge l'image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erreur lors de la lecture de l'image : {img_path}")
                continue

            # Redimensionne l'image
            resized_img = cv2.resize(img, target_size)

            # Enregistre l'image redimensionnée
            save_path = os.path.join(resized_folder, img_name)
            cv2.imwrite(save_path, resized_img)
            print(f"Image redimensionnée et enregistrée : {save_path}")

print("Redimensionnement terminé pour tous les dossiers.")
