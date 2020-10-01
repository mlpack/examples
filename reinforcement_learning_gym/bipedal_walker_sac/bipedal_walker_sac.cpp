#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/reinforcement_learning/sac.hpp>
#include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/reinforcement_learning/environment/env_type.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

// Used to run the agent on gym's environment (provided externally) for testing.
#include "../gym/environment.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;
using namespace gym;

int main()
{
  // Initializing the agent

  // Set up the state and action space.
  ContinuousActionEnv::State::dimension = 24;
  ContinuousActionEnv::Action::size = 4;

  // Set up the actor and critic networks.
  FFN<EmptyLoss<>, GaussianInitialization>
      policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.01));
  policyNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension, 128));
  policyNetwork.Add(new ReLULayer<>());
  policyNetwork.Add(new Linear<>(128, 128));
  policyNetwork.Add(new ReLULayer<>());
  policyNetwork.Add(new Linear<>(128, ContinuousActionEnv::Action::size));
  policyNetwork.Add(new TanHLayer<>());
  policyNetwork.ResetParameters();

  FFN<EmptyLoss<>, GaussianInitialization>
      qNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.01));
  qNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension + ContinuousActionEnv::Action::size, 128));
  qNetwork.Add(new ReLULayer<>());
  qNetwork.Add(new Linear<>(128, 128));
  qNetwork.Add(new ReLULayer<>());
  qNetwork.Add(new Linear<>(128, 1));
  qNetwork.ResetParameters();

  // Set up the replay method.
  RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

  // Set up training configurations.
  TrainingConfig config;
  config.ExplorationSteps() = 3200;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 1;

  /**In the cell below, we load a pretrained model by manually assigning values to 
   * the parameters of the network, after loading the parameters from their respective 
   * files `sac_q.txt` and `sac_policy.txt`.
   * The model was trained for 620 episodes.
   */
  arma::mat temp;
  data::Load("sac_q.txt", temp);
  qNetwork.Parameters() = temp.t();
  data::Load("sac_policy.txt", temp);
  policyNetwork.Parameters() = temp.t();

  // You can train the model from scratch by running the following:
  // c++
  // Set up Soft actor-critic agent.
  SAC<ContinuousActionEnv, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  const std::string environment = "BipedalWalker-v3";
  const std::string host = "gym.kurg.org";
  const std::string port = "4040";

  Environment env(host, port, environment);

  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;
  size_t consecutiveEpisodesTest = 50;
  while (true)
  {
    double episodeReturn = 0;
    env.reset();
    size_t steps = 0;
    do
    {
      agent.State().Data() = env.observation;
      agent.SelectAction();
      arma::mat action = {agent.Action().action};

      env.step(action);
      ContinuousActionEnv::State nextState;
      nextState.Data() = env.observation;

      replayMethod.Store(agent.State(), agent.Action(), env.reward, nextState, env.done, 0.99);
      episodeReturn += env.reward;
      agent.TotalSteps()++;
      steps++;
      if (agent.Deterministic() || agent.TotalSteps() < config.ExplorationSteps())
          continue;
      for (size_t i = 0; i < config.UpdateInterval(); i++)
          agent.Update();
    } while (!env.done);
    returnList.push_back(episodeReturn);
    episodes += 1;

    if (returnList.size() > consecutiveEpisodesTest)
        returnList.erase(returnList.begin());

    double averageReturn = std::accumulate(returnList.begin(),
                                           returnList.end(), 0.0) /
                           returnList.size();

    std::cout <<"Average return in last" << returnList.size()
              <<" consecutive episodes:" << averageReturn
              <<" steps:" << steps
              <<" Episode return:" << episodeReturn << std::endl;

    if (episodes % 10 == 0)
    {
        data::Save("./" + std::to_string(episodes) + "qNetwork.xml", "episode_" + std::to_string(episodes), qNetwork);
        data::Save("./" + std::to_string(episodes) + "policyNetwork.xml", "episode_" + std::to_string(episodes), policyNetwork);
    }
    if (averageReturn > -50)
        break;
  }

  // Testing the trained agent
  // It is so amazing to see how just a matrix of numbers, operated in a certain fashion, is able to develop a walking gait. 
  // Thats the beauty of Artificial Neural Networks!
  // Set up Soft actor-critic agent.

  // SAC<ContinuousActionEnv, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
  //     agent(config, qNetwork, policyNetwork, replayMethod);

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
    // State from the environment is passed to the agent's internal representation.
    agent.State().Data() = envTest.observation;

    // With the given state, the agent selects an action according to its defined policy.
    agent.SelectAction();

    // Action to take, decided by the policy.
    arma::mat action = {agent.Action().action};

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
//  auto video = xw::video_from_url(url).finalize();
}

