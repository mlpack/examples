//Including necessary libraries and namespaces
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/reinforcement_learning/sac.hpp>
#include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/reinforcement_learning/environment/env_type.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

// Used to run the agent on gym's environment (provided externally) for testing.
#include "cpp/environment.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
void train(gym::Environment& env,
           SAC<EnvironmentType, NetworkType, UpdaterType, PolicyType>& agent,
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
    env.reset();
    do
    {
      agent.State().Data() = env.observation;
      agent.SelectAction();
      arma::mat action = {double(agent.Action().action[0] * 2)};

      env.step(action);
      ContinuousActionEnv::State nextState;
      nextState.Data() = env.observation;

      replayMethod.Store(agent.State(), agent.Action(), env.reward, nextState,
          env.done, 0.99);
      episodeReturn += env.reward;
      agent.TotalSteps()++;
      if (agent.Deterministic() || agent.TotalSteps() < config.ExplorationSteps())
        continue;
      for (size_t i = 0; i < config.UpdateInterval(); i++)
        agent.Update();
    } while (!env.done);
    returnList.push_back(episodeReturn);
    episodes += 1;

    if (returnList.size() > consecutiveEpisodes)
      returnList.erase(returnList.begin());
        
    double averageReturn = std::accumulate(returnList.begin(),
                                           returnList.end(), 0.0) /
                           returnList.size();
    if (episodes % 4 == 0)
    {
      std::cout << "Avg return in last " << returnList.size()
          << " episodes: " << averageReturn
          << "\\t Episode return: " << episodeReturn
          << "\\t Total steps: " << agent.TotalSteps() << std::endl;
    }
  }
}

int main()
{
  //Initializing the agent

  // Set up the state and action space.
  ContinuousActionEnv::State::dimension = 3;
  ContinuousActionEnv::Action::size = 1;

  // Set up the actor and critic networks.
  FFN<EmptyLoss<>, GaussianInitialization>
      policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension, 32));
  policyNetwork.Add(new ReLULayer<>());
  policyNetwork.Add(new Linear<>(32, ContinuousActionEnv::Action::size));
  policyNetwork.Add(new TanHLayer<>());

  FFN<EmptyLoss<>, GaussianInitialization>
      qNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension +
                            ContinuousActionEnv::Action::size, 32));
  qNetwork.Add(new ReLULayer<>());
  qNetwork.Add(new Linear<>(32, 1));

  // Set up the policy method.
  RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

  // Set up training configurations.
  TrainingConfig config;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 1;

  // Set up Soft actor-critic agent.
  SAC<ContinuousActionEnv, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  // Preparation for training the agent

  // Set up the gym training environment.
  gym::Environment env("gym.kurg.org", "4040", "Pendulum-v0");

  // Initializing training variables.
  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;

  // The number of episode returns to keep track of.
  size_t consecutiveEpisodes = 25;

  // Function to train the agent on the Pendulum gym environment.
  //Let the training begin

  // Training the agent for a total of at least 5000 steps.
  train(env, agent, replayMethod, config, returnList, episodes, 
      consecutiveEpisodes, 5000);

  //Testing the trained agent
  agent.Deterministic() = true;

  // Creating and setting up the gym environment for testing.
  gym::Environment envTest("gym.kurg.org", "4040", "Pendulum-v0");
  envTest.monitor.start("./dummy/", true, true);

  // Resets the environment.
  envTest.reset();
  envTest.render();

  double totalReward = 0;
  size_t totalSteps = 0;

  // Testing the agent on gym's environment.
  while (1)
  {
    // State from the environment is passed to the agent's internal representation.
    agent.State().Data() = envTest.observation;

    // With the given state, the agent selects an action according to its defined policy.
    agent.SelectAction();

    // Action to take, decided by the policy.
    arma::mat action = {double(agent.Action().action[0] * 2)};

    envTest.step(action);
    totalReward += envTest.reward;
    totalSteps += 1;

    if (envTest.done)
    {
      std::cout << " Total steps: " << totalSteps << "\\t Total reward: "
          << totalReward << std::endl;
      break;
    }

    // Uncomment the following lines to see the reward and action in each step.
    // std::cout << " Current step: " << totalSteps << "\\t current reward: "
    //   << totalReward << "\\t Action taken: " << action;
  }

  envTest.close();
  std::string url = envTest.url();
  std::cout << url;

  // A little more training...
  // Training the same agent for a total of at least 40000 steps.
  train(env, agent, replayMethod, config, returnList, episodes, 
      consecutiveEpisodes, 40000);

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
  while (1)
  {
    // State from the environment is passed to the agent's internal representation.
    agent.State().Data() = envTest.observation;

    // With the given state, the agent selects an action according to its defined policy.
    agent.SelectAction();

    // Action to take, decided by the policy.
    arma::mat action = {double(agent.Action().action[0] * 2)};

    envTest.step(action);
    totalReward += envTest.reward;
    totalSteps += 1;

    if (envTest.done)
    {
      std::cout << " Total steps: " << totalSteps << "\\t Total reward: "
          << totalReward << std::endl;
      break;
    }

    // Uncomment the following lines to see the reward and action in each step.
    // std::cout << " Current step: " << totalSteps << "\\t current reward: "
    //   << totalReward << "\\t Action taken: " << action;
  }

  envTest.close();
  url = envTest.url();
  std::cout << url;
}
