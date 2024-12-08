import os

# Dossier contenant vos fichiers
dossier = "C:\\Users\\Loren\\anaconda3\\envs\\pyppbox\\Lib\\site-packages\\pyppbox\\data\\datasets\\TEST_GROUP\\body_128x256\\max_v2"

# Parcourir tous les fichiers du dossier
for fichier in os.listdir(dossier):
    chemin_complet = os.path.join(dossier, fichier)
    
    # Vérifie si c'est un fichier
    if os.path.isfile(chemin_complet):
        base, extension = os.path.splitext(fichier)
        nouveau_nom = f"{base}bis{extension}"
        os.rename(chemin_complet, os.path.join(dossier, nouveau_nom))

print("Tous les fichiers ont été renommés avec 'bis' !")
