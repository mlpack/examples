/**
 * This example shows how to get use 3-Step Double DQN with Prioritized Replay
 * to train an agent to get high scores for the
 * [Acrobot](https://gym.openai.com/envs/Acrobot-v1) environment.
 * We make the agent train and test on OpenAI Gym toolkit's GUI interface provided
 * through a distributed infrastructure (TCP API). More details can be found
 * [here](https://github.com/zoq/gym_tcp_api).
 * A video of the trained agent can be seen in the end.
 */

// Including necessary libraries and namespaces.
#include <mlpack.hpp>

// Used to run the agent on gym's environment (provided externally) for testing.
#include "../gym/environment.hpp"

using namespace mlpack;
using namespace ens;

// Function to train the agent on the Acrobot-v1 gym environment.
template<typename EnvironmentType,
         typename NetworkType,
         typename UpdaterType,
         typename PolicyType,
         typename ReplayType = RandomReplay<EnvironmentType>>
void Train(gym::Environment& env,
           QLearning<EnvironmentType,
                     NetworkType,
                     UpdaterType,
                     PolicyType,
                     ReplayType>& agent,
           PrioritizedReplay<EnvironmentType>& replayMethod,
           TrainingConfig& config,
           std::vector<double>& returnList,
           size_t& episodes,
           size_t& consecutiveEpisodes,
           const size_t numSteps)
{
  agent.Deterministic() = false;
  std::cout << "Training for " << numSteps << " steps." << std::endl;
  while (agent.TotalSteps() < numSteps)
  {
    double episodeReturn = 0;
    env.reset();
    do
    {
      agent.State().Data() = env.observation;
      agent.SelectAction();
      arma::mat action = {double(agent.Action().action)};

      env.step(action);
      DiscreteActionEnv::State nextState;
      nextState.Data() = env.observation;

      replayMethod.Store(
          agent.State(), agent.Action(), env.reward, nextState, env.done, 0.99);
      episodeReturn += env.reward;
      agent.TotalSteps()++;
      if (agent.Deterministic()
          || agent.TotalSteps() < config.ExplorationSteps())
      {
        continue;
      }
      agent.TrainAgent();
    } while (!env.done);
    returnList.push_back(episodeReturn);
    episodes += 1;

    if (returnList.size() > consecutiveEpisodes)
      returnList.erase(returnList.begin());

    double averageReturn =
        std::accumulate(returnList.begin(), returnList.end(), 0.0)
        / returnList.size();
    if (episodes % 5 == 0)
    {
      std::cout << "Avg return in last " << consecutiveEpisodes
                << " episodes: " << averageReturn
                << "\t Episode return: " << episodeReturn
                << "\t Total steps: " << agent.TotalSteps() << std::endl;
    }
  }
}

int main()
{
  // Initializing the agent.
  // Set up the state and action space.
  DiscreteActionEnv::State::dimension = 6;
  DiscreteActionEnv::Action::size = 3;

  // Set up the network.
  FFN<MeanSquaredError, GaussianInitialization> module(
      MeanSquaredError(), GaussianInitialization(0, 1));
  module.Add<Linear>(DiscreteActionEnv::State::dimension, 64);
  module.Add<ReLULayer>();
  module.Add<Linear>(64, DiscreteActionEnv::Action::size);
  SimpleDQN<> model(module);

  // Set up the policy method.
  GreedyPolicy<DiscreteActionEnv> policy(1.0, 1000, 0.1, 0.99);
  // To enable 3-step learning, we set the last parameter of the replay method as 3.
  PrioritizedReplay<DiscreteActionEnv> replayMethod(64, 5000, 0.6, 3);

  // Set up training configurations.
  TrainingConfig config;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 500;

  // We use double Q learning for this example.
  config.DoubleQLearning() = true;

  // Set up DQN agent.
  QLearning<DiscreteActionEnv,
            decltype(model),
            AdamUpdate,
            decltype(policy),
            decltype(replayMethod)>
      agent(config, model, policy, replayMethod);

  // Preparation for training the agent
  // Set up the gym training environment.
  gym::Environment env("gym.kurg.org", "4040", "Acrobot-v1");

  // Initializing training variables.
  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;

  // The number of episode returns to keep track of.
  size_t consecutiveEpisodes = 50;

  // Training the agent for a total of at least 25000 steps.
  Train(env,
        agent,
        replayMethod,
        config,
        returnList,
        episodes,
        consecutiveEpisodes,
        25000);

  // Testing the trained agent
  agent.Deterministic() = true;

  // Creating and setting up the gym environment for testing.
  gym::Environment envTest("gym.kurg.org", "4040", "Acrobot-v1");
  envTest.monitor.start("./dummy/", true, true);

  // Resets the environment.
  envTest.reset();
  envTest.render();

  double totalReward = 0;
  size_t totalSteps = 0;

  // Testing the agent on gym's environment.
  while (true)
  {
    // State from the environment is passed to the agent's internal representation.
    agent.State().Data() = envTest.observation;

    // With the given state, the agent selects an action according to its defined policy.
    agent.SelectAction();

    // Action to take, decided by the policy.
    arma::mat action = {double(agent.Action().action)};

    envTest.step(action);
    totalReward += envTest.reward;
    totalSteps += 1;

    if (envTest.done)
    {
      std::cout << " Total steps: " << totalSteps
                << "\t Total reward: " << totalReward << std::endl;
      break;
    }

    // Uncomment the following lines to see the reward and action in each step.
    // std::cout << " Current step: " << totalSteps << "\t current reward: "
    //   << totalReward << "\t Action taken: " << action;
  }

  envTest.close();
  std::cout << envTest.url() << std::endl;

  /** 
   * Due to the stochasticity of the environment, it's quite possible that sometimes
   * the agent is not able to solve it in each test. So, we test the agent once
   * more, just to be sure.
   */
  agent.Deterministic() = true;

  // Creating and setting up the gym environment for testing.
  envTest.monitor.start("./dummy/", true, true);

  // Resets the environment.
  envTest.reset();
  envTest.render();

  totalReward = 0;
  totalSteps = 0;

  // Testing the agent on gym's environment.
  while (true)
  {
    // State from the environment is passed to the agent's internal representation.
    agent.State().Data() = envTest.observation;

    // With the given state, the agent selects an action according to its defined policy.
    agent.SelectAction();

    // Action to take, decided by the policy.
    arma::mat action = {double(agent.Action().action)};

    envTest.step(action);
    totalReward += envTest.reward;
    totalSteps += 1;

    if (envTest.done)
    {
      std::cout << " Total steps: " << totalSteps
                << "\t Total reward: " << totalReward << std::endl;
      break;
    }

    // Uncomment the following lines to see the reward and action in each step.
    // std::cout << " Current step: " << totalSteps << "\t current reward: "
    //   << totalReward << "\t Action taken: " << action;
  }

  envTest.close();
  std::cout << envTest.url() << std::endl;
}
