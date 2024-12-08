#################################################################################
# Example of training classifier of a reider -> `trainReIDClassifier()`
#################################################################################

from pyppbox.standalone import trainReIDClassifier


# Reider
    myreider={
        'ri_name': 'Torchreid', 
        'classifier_pkl': r'C:\Users\Loren\anaconda3\envs\pyppbox\Lib\site-packages\pyppbox\data\modules\torchreid\classifier\test4.pkl', 
        'train_data': r'C:/Users\Loren\anaconda3\envs\pyppbox\Lib\site-packages\pyppbox\data\datasets\TEST_GROUP\body_128x256', 
        'model_name': 'osnet_ain_x1_0', 
        'model_path': r'C:\Users\Loren\anaconda3\envs\pyppbox\Lib\site-packages\pyppbox\data\modules\torchreid\models\torchreid\osnet_ain_ms_d_c.pth.tar', 
        'min_confidence': 0.35,
        'device': 'cuda'
    }

trainReIDClassifier(
    reider=myreider, 
    train_data="", # Set train_data="" means using the default 'train_data' in line 12
    classifier_pkl="" # Set classifier_pkl="" to use the default in line 14
)

