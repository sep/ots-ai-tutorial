# Adapted from https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py

from collections import defaultdict
import numpy as np
from tqdm import tqdm
import gymnasium as gym


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
        if np.random.random() < self.epsilon:
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
        temporalDifference = (
            reward + self.futureDiscount * futureQValue - self.qValues[stateKey][action]
        )

        self.qValues[stateKey][action] = (
            self.qValues[stateKey][action] + self.learningRate * temporalDifference
        )
        self.errors.append(temporalDifference)

    def decay_epsilon(self):
        self.epsilon = max(self.stopEpsilon, self.epsilon - decay)

        
def cartPoleHash(observation):
    # cart position, cart velocity, pole angle, pole angular velocity
    # Exercise 1: What happens when we change this?
    #             Can we discard elements?  Reduce their resolution?
    return (observation[0],observation[1],observation[2],observation[3])

def mountainCarHash(observation):
    # Exercise 2: What should this be? How do we figure it out?
    # Exercise 3: As above, what about reducing the space.  How would you do so effectively?
    # car position, velocity
    return (observation[0], observation[1])

def trainAgent(env, agent, numEpisodes, decay):
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
            done = terminated or truncated
            obs = nextObservation

            agent.decay_epsilon()

def buildEnvAgent(
        domain, hash, renderMode=None, learningRate = 0.01,
        numEpisodes = 100_000, startEpsilon = 1.0, stopEpsilon = 0.1, futureDiscount = 0.95):
    None
    environemnt = None
    if renderMode is None:
        environment = gym.make(domain)
    else:
        environment = gym.make(domain, render_mode=renderMode)
    decay = startEpsilon / (numEpisodes / 2)  # reduce the exploration over time
    env = gym.wrappers.RecordEpisodeStatistics(environment, deque_size=numEpisodes)
    agent = QLearningAgent(
        environment = env,
        hash = hash,
        learningRate = learningRate,
        startEpsilon = startEpsilon,
        decay = decay,
        stopEpsilon = stopEpsilon,
        futureDiscount = futureDiscount
    )
    return (env, agent, numEpisodes, decay)


if __name__ == "__main__":
    # Exercise 0: Run an agent or two with the human display mode, just to see what it's doing

    #env,agent,episodes,decay = buildEnvAgent('CartPole-v1', cartPoleHash, renderMode='human')
    #env,agent,episodes,decay = buildEnvAgent('MountainCar-v0', mountainCarHash, renderMode='human')

    # Exercise 1: Run an agent (without display) to completion to get
    # a sense for how computationally intense these processes are.
    env,agent,episodes,decay = buildEnvAgent('CartPole-v1', cartPoleHash)
    #env,agent,episodes,decay = buildEnvAgent('MountainCar-v0', mountainCarHash)

    

    trainAgent(env,agent,episodes,decay)
