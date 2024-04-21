import gymnasium as gym
from ddqn_agent import DDQNAgent



env = gym.make('CartPole-v1')
lr = 0.001
action_n = 2
input_n = 4
discount = 0.95 
epsilon_decay = 0.975
agent = DDQNAgent(lr, action_n, input_n, discount, epsilon_decay)

epochs = 150
for e in range(epochs):
    score = 0
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.action(state) 
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            reward = -1

        agent.get_memory(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        
        score += reward

    if e % 5:
        agent.reset()

    print(f'epoch: {e}/{epochs} score: {score} epsilon: {agent.epsilon}')
    agent.epsilon_change()


env.close()

