/**
 * In this example, we show how to use a simple DQN to train an agent to solve the 
 * [MountainCar](https://gym.openai.com/envs/MountainCar-v0) environment.
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

template<typename EnvironmentType,
         typename NetworkType,
         typename UpdaterType,
         typename PolicyType,
         typename ReplayType = RandomReplay<EnvironmentType>>
void Train(
    gym::Environment& env,
    QLearning<EnvironmentType, NetworkType, UpdaterType, PolicyType>& agent,
    RandomReplay<EnvironmentType>& replayMethod,
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
    double adjustedEpisodeReturn = 0;
    env.reset();
    do
    {
      agent.State().Data() = env.observation;
      agent.SelectAction();
      arma::mat action = {double(agent.Action().action)};

      env.step(action);
      DiscreteActionEnv::State nextState;
      nextState.Data() = env.observation;

      // Use an adjusted reward for task completion.
      double adjustedReward = env.reward;
      if (nextState.Data()[0] < -0.8)
        adjustedReward += 0.5;

      replayMethod.Store(agent.State(),
                         agent.Action(),
                         adjustedReward,
                         nextState,
                         env.done,
                         0.99);
      episodeReturn += env.reward;
      adjustedEpisodeReturn += adjustedReward;
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
                << "\t Adjusted return: " << adjustedEpisodeReturn
                << "\t Total steps: " << agent.TotalSteps() << std::endl;
    }
  }
}

int main()
{
  // Initializing the agent.
  // Set up the state and action space.
  DiscreteActionEnv::State::dimension = 2;
  DiscreteActionEnv::Action::size = 3;

  // Set up the network.
  FFN<MeanSquaredError, GaussianInitialization> network(
      MeanSquaredError(), GaussianInitialization(0, 1));
  network.Add<Linear>(128);
  network.Add<ReLULayer>();
  network.Add<Linear>(DiscreteActionEnv::Action::size);
  // Set up the network.
  SimpleDQN<> model(network);

  // Set up the policy method.
  GreedyPolicy<DiscreteActionEnv> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<DiscreteActionEnv> replayMethod(32, 10000);

  // Set up training configurations.
  TrainingConfig config;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 400;

  // Set up DQN agent.
  QLearning<DiscreteActionEnv,
            decltype(model),
            AdamUpdate,
            decltype(policy),
            decltype(replayMethod)>
      agent(config, model, policy, replayMethod);

  // Preparation for training the agent.

  // Set up the gym training environment.
  gym::Environment env("gym.kurg.org", "4040", "MountainCar-v0");

  // Initializing training variables.
  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;

  // The number of episode returns to keep track of.
  size_t consecutiveEpisodes = 50;

  /**
   * An important point to note for Mountain Car setup is that for each step that
   * the car does not reach the goal located at position `0.5`, the environment
   * returns a reward of `-1`. Now, since the agent’s reward never changes until 
   * completion of the episode, it is difficult for our algorithm to improve until
   * it randomly reaches the top of the hill.
   * That is unless we modify the reward by giving an additional `0.5` reward for
   * every time the agent managed to drag the car in the backward direction 
   * (i.e position < `-0.8`). This was important to gain momentum to climb the hill.
   * This minor tweak can greatly increase sample efficiency.
   * Note here that `Episode return:` is the actual (environment's) return, whereas 
   * `Adjusted return:` is the return calculated from the adjusted reward function 
   * we described earlier.
   */

  // Training the agent for a total of at least 75 episodes.
  Train(env,
        agent,
        replayMethod,
        config,
        returnList,
        episodes,
        consecutiveEpisodes,
        200 * 75);

  // Testing the trained agent
  agent.Deterministic() = true;

  // Creating and setting up the gym environment for testing.
  gym::Environment envTest("gym.kurg.org", "4040", "MountainCar-v0");
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

  // A little more training...
  // Training the same agent for a total of at least 300 episodes.
  Train(env,
        agent,
        replayMethod,
        config,
        returnList,
        episodes,
        consecutiveEpisodes,
        200 * 300);

  // Final agent testing!
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
