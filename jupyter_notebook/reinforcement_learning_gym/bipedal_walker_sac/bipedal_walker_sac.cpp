/**
 * Here, we train a [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) agent to get
 * high scores for the [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) 
 * environment.
 * We make the agent train and test on OpenAI Gym toolkit's GUI interface provided 
 * through a distributed infrastructure (TCP API). More details can be found
 * [here](https://github.com/zoq/gym_tcp_api).
 * A video of the trained agent can be seen in the end.
 */

// Including necessary libraries and namespaces
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

// Used to run the agent on gym's environment (provided externally) for testing.
#include "../gym/environment.hpp"

using namespace mlpack;
using namespace ens;
using namespace gym;

template<typename EnvironmentType,
         typename NetworkType,
         typename UpdaterType,
         typename PolicyType,
         typename ReplayType = RandomReplay<EnvironmentType>>
void Train(gym::Environment& env,
           SAC<EnvironmentType, NetworkType, UpdaterType, PolicyType>& agent,
           NetworkType qNetwork,
           NetworkType policyNetwork,
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
      arma::mat action = {agent.Action().action};

      env.step(action);
      ContinuousActionEnv::State nextState;
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
    } while (!env.done);
    returnList.push_back(episodeReturn);

    episodes += 1;

    if (returnList.size() > consecutiveEpisodes)
      returnList.erase(returnList.begin());

    double averageReturn =
        std::accumulate(returnList.begin(), returnList.end(), 0.0)
        / returnList.size();

    std::cout << "Average return in last" << returnList.size()
              << " consecutive episodes:" << averageReturn
              << " steps:" << agent.TotalSteps()
              << " Episode return:" << episodeReturn << std::endl;

    if (episodes % 10 == 0)
    {
      data::Save("./" + std::to_string(episodes) + "qNetwork.xml",
          "episode_" + std::to_string(episodes), qNetwork);
      data::Save("./" + std::to_string(episodes) + "policyNetwork.xml",
          "episode_" + std::to_string(episodes), policyNetwork);
    }
    if (averageReturn > -50)
      break;
  }
}

int main()
{
  // Initializing the agent.
  // Set up the state and action space.
  ContinuousActionEnv::State::dimension = 24;
  ContinuousActionEnv::Action::size = 4;

  bool usePreTrainedModel = true;

  // Set up the actor and critic networks.
  FFN<EmptyLoss, GaussianInitialization> policyNetwork(
      EmptyLoss(), GaussianInitialization(0, 0.01));
  policyNetwork.Add<Linear>(128);
  policyNetwork.Add<ReLU>();
  policyNetwork.Add<Linear>(128);
  policyNetwork.Add<ReLU>();
  policyNetwork.Add<Linear>(ContinuousActionEnv::Action::size);
  policyNetwork.Add<TanH>();
  policyNetwork.ResetParameters();

  FFN<EmptyLoss, GaussianInitialization> qNetwork(
      EmptyLoss(), GaussianInitialization(0, 0.01));
  qNetwork.Add<Linear>(128);
  qNetwork.Add<ReLU>();
  qNetwork.Add<Linear>(128);
  qNetwork.Add<ReLU>();
  qNetwork.Add<Linear>(1);
  qNetwork.ResetParameters();

  // Set up the replay method.
  RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

  // Set up training configurations.
  TrainingConfig config;
  config.ExplorationSteps() = 3200;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 1;

  /**
   * We load a pretrained model by manually assigning values to
   * the parameters of the network, after loading the parameters from their
   * respective files `sac_q.txt` and `sac_policy.txt`. The model was trained
   * for 620 episodes.
   */
  arma::mat temp;
  data::Load("sac_q.txt", temp);
  qNetwork.Parameters() = temp.t();
  data::Load("sac_policy.txt", temp);
  policyNetwork.Parameters() = temp.t();

  /**
   * You can train the model from scratch by running the following
   * training function or you can use the pretrained model already
   * provided in this repository.
   * To default is to use the usePreTrainedModel. Otherwise you can disable this
   * by change the usePreTrainedModel to false and then recompile this example.
   */
  SAC<ContinuousActionEnv,
      decltype(qNetwork),
      decltype(policyNetwork),
      AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  const std::string environment = "BipedalWalker-v3";
  const std::string host = "gym.kurg.org";
  const std::string port = "4040";

  Environment env(host, port, environment);

  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;
  size_t consecutiveEpisodes = 50;
  if (!usePreTrainedModel)
  {
    Train(env,
          agent,
          qNetwork,
          policyNetwork,
          replayMethod,
          config,
          returnList,
          episodes,
          consecutiveEpisodes,
          5000);
  }

  /**
   * Testing the trained agent.
   * It is so amazing to see how just a matrix of numbers, operated in a certain
   * fashion, is able to develop a walking gait. Thats the beauty of Artificial
   * Neural Networks! Set up Soft actor-critic agent.
   */
  agent.Deterministic() = true;

  // Creating and setting up the gym environment for testing.
  gym::Environment envTest("gym.kurg.org", "4040", "BipedalWalker-v3");
  envTest.monitor.start("./dummy/", true, true);

  // Resets the environment.
  envTest.reset();
  envTest.render();

  double totalReward = 0;
  size_t totalSteps = 0;

  // Testing the agent on gym's environment.
  while (1)
  {
    // State from the environment is passed to the agent's internal
    // representation.
    agent.State().Data() = envTest.observation;

    // With the given state, the agent selects an action according to its
    // defined policy.
    agent.SelectAction();

    // Action to take, decided by the policy.
    arma::mat action = {agent.Action().action};

    envTest.step(action);
    totalReward += envTest.reward;
    totalSteps += 1;

    if (envTest.done)
    {
      std::cout << " Total steps: " << totalSteps
                << "\\t Total reward: " << totalReward << std::endl;
      break;
    }

    // Uncomment the following lines to see the reward and action in each step.
    // std::cout << " Current step: " << totalSteps << "\\t current reward: "
    //   << totalReward << "\\t Action taken: " << action;
  }

  envTest.close();
  std::cout<< envTest.url() << std::endl;
}
