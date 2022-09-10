import retro
import cv2

from src.processing.preprocessing import fetch_region_pipeline

if __name__ == '__main__':
    env = retro.make(game='SuperMarioKart-Snes', state='states/1P_DK_Shroom_Solo')
    env.reset()

    pre_pipeline = fetch_region_pipeline(cropper=True, cropper_roi=[0, 112, 0, 256])

    for _ in range(1000):
        state, reward, done, info = env.step(env.action_space.sample()) # take a random action
        state = pre_pipeline.transform(state)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', state)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    env.close()

    print(state.shape, info)