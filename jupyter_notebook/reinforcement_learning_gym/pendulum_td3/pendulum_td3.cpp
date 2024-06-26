/**
 * In this example, we train a 
 * [Twin Delayed Deep Deterministic Policy Gradient]
 * (https://arxiv.org/abs/1802.09477)
 * agent to get high scores for the
 * [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) 
 * environment.
 * We make the agent train and test on OpenAI Gymnasium toolkit's GUI interface 
 * provided through a distributed infrastructure (TCP API). More details can be 
 * found [here](https://github.com/zoq/gym_tcp_api).
 * A video of the trained agent can be seen in the end.
 */

// Including necessary libraries and namespaces.
#include <mlpack.hpp>

// Used to run the agent on gym's environment for testing.
#include "../gym/environment.hpp"

using namespace mlpack;
using namespace ens;

template<typename EnvironmentType,
         typename NetworkType,
         typename UpdaterType,
         typename PolicyType,
         typename ReplayType = RandomReplay<EnvironmentType>>
void Train(gym::Environment& env,
           TD3<EnvironmentType, NetworkType, UpdaterType, PolicyType>& agent,
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
      ContinuousActionEnv<3, 1>::State nextState;
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

      for (size_t i = 0; i < config.UpdateInterval(); i++)
        agent.Update();
    }

    while (!env.done);
    returnList.push_back(episodeReturn);

    episodes += 1;

    if (returnList.size() > consecutiveEpisodes)
      returnList.erase(returnList.begin());

    double averageReturn =
        std::accumulate(returnList.begin(), returnList.end(), 0.0)
        / returnList.size();
    if (episodes % 4 == 0)
    {
      std::cout << "Avg return in last " << returnList.size()
                << " episodes: " << averageReturn
                << "\t Episode return: " << episodeReturn
                << "\t Total steps: " << agent.TotalSteps() << std::endl;
    }
  }
}

int main()
{
  // Initializing the agent.

  // Set up the actor and critic networks.
  FFN<EmptyLoss, GaussianInitialization> policyNetwork(
      EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add<Linear>(32);
  policyNetwork.Add<ReLU>();
  policyNetwork.Add<Linear>(ContinuousActionEnv<3, 1>::Action::size);
  policyNetwork.Add<TanH>();

  FFN<EmptyLoss, GaussianInitialization> qNetwork(
      EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add<Linear>(32);
  qNetwork.Add<ReLU>();
  qNetwork.Add<Linear>(1);

  // Set up the policy method.
  RandomReplay<ContinuousActionEnv<3, 1>> replayMethod(32, 10000);

  // Set up training configurations.
  TrainingConfig config;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 1;
  
  // Set up the TD3 agent.
  TD3<ContinuousActionEnv<3, 1>,
      decltype(qNetwork),
      decltype(policyNetwork),
      AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  // Preparation for training the agent.

  // Set up the gym training environment.
  gym::Environment env("localhost", "4040", "Pendulum-v1");

  // Initializing training variables.
  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;

  // The number of episode returns to keep track of.
  size_t consecutiveEpisodes = 25;

  // Training the agent for a total of 50000 steps.
  Train(env,
        agent,
        replayMethod,
        config,
        returnList,
        episodes,
        consecutiveEpisodes,
        50000);

  // Testing the trained agent.
  agent.Deterministic() = true;

  // Creating and setting up the gym environment for testing.
  gym::Environment envTest("localhost", "4040", "Pendulum-v1-render");

  // Resets the environment.
  envTest.reset();

  double totalReward = 0;
  size_t totalSteps = 0;

  while (true)
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
      std::cout << " Total steps: " << totalSteps
                << "\t Total reward: " << totalReward << std::endl;
      break;
    }
  }

  envTest.close();
  std::cout << envTest.url() << std::endl;
}
