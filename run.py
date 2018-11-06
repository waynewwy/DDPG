
from CourseEnv import env
from rl import DDPG
import pylab as pl

MAX_EPISODES = 1000
MAX_EP_STEPS = 100
ON_TRAIN = True


# set env
env = env()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.

        for j in range(MAX_EP_STEPS):
            # env.render()
            x = []
            y = []
            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
            s = s_
            # x = j
            # y = s[1]
            # pl.plot(x, y, 'or')
            j += 1

            if done or j == MAX_EP_STEPS-1:
                x.append(i)
                y.append(ep_r)
                pl.plot(x, y, 'or')
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    pl.show()  # show the plot on the screen
    rl.save()

def eval():
    rl.restore()
    s = env.reset()
    while True:
        a = rl.choose_action(s)
        s, r, done = env.step(a)


if ON_TRAIN:
    train()
else:
    eval()



