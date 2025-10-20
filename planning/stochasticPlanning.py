# Adapted from https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py

from collections import defaultdict
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(
        self,
        environment,
        hash,
        learningRate,
        startEpsilon,
        decay,
        stopEpsilon,
        futureDiscount
    ):
        """
        Args:
            learningRate: The learning rate
            startEpsilon: The initial epsilon value
            decay: The decay for epsilon
            stopEpsilon: The final epsilon value
            futureDiscount: The discount factor for computing the Q-value
        """
        self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learningRate = learningRate
        self.futureDiscount = futureDiscount
        self.environment = environment
        # The hash is how the solver maps domain states into dictionaries
        # Changing the hash changes everything
        self.hash = hash
        self.epsilon = startEpsilon
        self.decay = decay
        self.stopEpsilon = stopEpsilon

        self.errors = []

    def getAction(self, observation):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        stateKey = self.hash(observation)
        # with probability epsilon return a random action to explore the environment
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return self.environment.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.qValues[stateKey]))

    def update(
        self,
        observation,
        action,
        reward,
        terminated,
        nextObservation
    ):
        """Updates the Q-value of an action."""
        stateKey = self.hash(observation)
        nextStateKey = self.hash(nextObservation)
        futureQValue = (not terminated) * np.max(self.qValues[nextStateKey])
        temporalDifference = (reward + (self.futureDiscount * futureQValue)) - self.qValues[stateKey][action]
        self.qValues[stateKey][action] = (
            self.qValues[stateKey][action] + self.learningRate * temporalDifference
        )
        self.errors.append(temporalDifference)
        if False:  # uncomment if you'd like to see the gorey details of the updates
            print(stateKey, nextStateKey)
            print(not terminated, self.qValues[nextStateKey])
            print(reward, self.futureDiscount * futureQValue)
            print("td", temporalDifference, "act", action, "q", self.qValues[stateKey][action])
            print()
            print()

    def decay_epsilon(self):
        self.epsilon = max(self.stopEpsilon, self.epsilon - decay)


PRECISION=2
def cartPoleHash(observation):
    # Exercise 3: What happens when we change this?
    #             Can we discard elements?  Reduce their resolution?
    # cart position, cart velocity, pole angle, pole angular velocity
    
    return (
        round(observation[0],PRECISION),
        round(observation[1],PRECISION),
        round(observation[2],PRECISION),
        round(observation[3],PRECISION)
    )

def mountainCarHash(observation):
    # Exercise 4: As above, what about reducing the space.  How would you do so effectively?
    # car position, velocity
    return (observation[0], observation[1])

def trainAgent(env, agent, numEpisodes, decay, display=False):
    for episode in tqdm(range(numEpisodes)):
        obs, info = env.reset()
        done = False
        
        # play one episode
        while not done:
            env.render()
            action = agent.getAction(obs)
            nextObservation, reward, terminated, truncated, info = env.step(action)
            # update the agent
            agent.update(obs, action, reward, terminated, nextObservation)
            
            # update if the environment is done
            done = terminated or (truncated and (not display))
            obs = nextObservation

            agent.decay_epsilon()

def buildEnvAgent(
        domain, hash, renderMode=None, learningRate = 0.1,
        numEpisodes = 100_000, startEpsilon = 1.0, stopEpsilon = 0., futureDiscount = 0.95):
    None
    if renderMode is None:
        environment = gym.make(domain)
    else:
        environment = gym.make(domain, render_mode=renderMode)
    visEnv = gym.make(domain, render_mode='human')
    decay = startEpsilon / (numEpisodes / 2)  # reduce the exploration over time
    env = gym.wrappers.RecordEpisodeStatistics(environment)#, deque_size=numEpisodes)
    agent = QLearningAgent(
        environment = env,
        hash = hash,
        learningRate = learningRate,
        startEpsilon = startEpsilon,
        decay = decay,
        stopEpsilon = stopEpsilon,
        futureDiscount = futureDiscount
    )
    return (env, agent, numEpisodes, decay, visEnv)


def visualizeRewardOverTime(env, agent):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        ) / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    ) / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.errors), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Exercise 0: Run an agent or two with the human display mode, just to see what it's doing

    #env,agent,episodes,decay,visEnv = buildEnvAgent('CartPole-v1', cartPoleHash, renderMode='human')
    #env,agent,episodes,decay,visEnv = buildEnvAgent('MountainCar-v0', mountainCarHash, renderMode='human')

    # Exercise 1: Run an agent (without display) to completion to get
    # a sense for how computationally intense these processes are.
    env,agent,episodes,decay,visEnv = buildEnvAgent('CartPole-v1', cartPoleHash, numEpisodes=1_000_000)
    #env,agent,episodes,decay,visEnv = buildEnvAgent('MountainCar-v0', mountainCarHash)

    trainAgent(env,agent,episodes,decay)
    visualizeRewardOverTime(env,agent)
    trainAgent(visEnv,agent, 1, decay, display=True)
