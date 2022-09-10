import json
import pickle

import cv2
import pandas as pd
import retro
from gym.wrappers import Monitor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.processing.preprocessing import fetch_region_pipeline
from src.utils import load_config

def train_model(data_path, model_path):
    # Loading data
    merged_df = pd.read_csv(data_path)
    merged_df = merged_df.drop(columns=['action'])
    
    X = merged_df.loc[:, merged_df.columns != 'label'].values
    y = merged_df.loc[:, 'label'].values

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Training a simple MLP
    clf = MLPClassifier(
        hidden_layer_sizes=(300,), 
        random_state=42, max_iter=500,
        validation_fraction=0.2, verbose=True).fit(X_train, y_train)

    clf.score(X_test, y_test)

    pickle.dump(clf, open(model_path, 'wb'))

# TODO: rename to class mapping
def get_action(model, state, output_mapping):
    output = str(int(model.predict([state])[0]))
    return eval(output_mapping[output])

if __name__ == '__main__':
    data_path = 'data/mnist_appoch/train_df.csv'
    config_path = 'configs/mnist_state_trasnform.json'
    model_path = 'models/mlp_scikit.pkl'  
    output_mapping_path = 'data/mnist_appoch/output_mapping.json'

    rec_folder = 'data/checkpoints'

    # train model
    # train_model(data_path, model_path)

    # load preprocessing config
    config = load_config(config_path)
    pre_pipeline = fetch_region_pipeline(**config['preprocessing'])

    # load model
    clf = pickle.load(open(model_path, 'rb'))

    # load output to actions mapping
    output_mapping = json.load(open(output_mapping_path, 'r'))

    env = retro.make(game='SuperMarioKart-Snes', state='states/1P_DK_Shroom_R1')
    env = Monitor(env, rec_folder, force=True, 
                    video_callable=lambda episode_id: True, )

    state = env.reset()
    done = False

    while not done:
        pre_state = pre_pipeline.transform(state)
        pre_state = pre_state.flatten() / 255.0

        action = get_action(clf, pre_state, output_mapping)

        state, reward, done, info = env.step(action) # take a random action
        
        state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', state)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    env.close()
