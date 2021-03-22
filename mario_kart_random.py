import retro
import cv2

env = retro.make(game='SuperMarioKart-Snes')
env.reset()

roi = [20, 70, 10, 190]

for _ in range(1000):
    state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    state = state[roi[0]:roi[1], roi[2]:roi[3]]
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
env.close()
print(state.shape, info)