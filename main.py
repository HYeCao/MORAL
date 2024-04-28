import gym
import d4rl
import numpy as np
import time
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config import args_setting
import utils.static
from module.ensemble_trainer import EnsembleTrainer
from module.adversarial_dynamics import AdversarialDynamics
from module.offline_agent import OfflineAgent
from module.replay_memory import ReplayMemory
from utils.utils import setup_seed, Evaluator

def tran_model(env):
    # get offline datasets
    dataset = d4rl.qlearning_dataset(env.unwrapped)
    state = dataset['observations']
    action = dataset['actions']
    next_state = dataset['next_observations']
    reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    done = np.expand_dims(np.squeeze(dataset['terminals']), 1)

    # define the state action space
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    reward_func = None
    done_func = utils.static[args.task.split('-')[0]].termination_fn

    # ensemble model training
    predict_reward = reward_func is None
    ensemble_trainer = EnsembleTrainer(state_dim, action_space.shape[0], predict_reward, args)
    ensemble_trainer.train({'obs': state, 'act': action, 'obs_next': next_state, 'rew': reward})
    transition = ensemble_trainer.transition

    # define replay memory
    real_memory = ReplayMemory(state.shape[0])
    normalized_reward = transition.get_normalized_reward(reward)
    # get replay buffer
    real_memory.push(state, action, normalized_reward, next_state, done)
    offline_agent = OfflineAgent(state_dim, action_space, real_memory, args, value_clip=predict_reward)

    all_state = np.concatenate([state, next_state[~done.astype(bool).reshape([-1])]], axis=0)
    adv_dyna = AdversarialDynamics(all_state, offline_agent, transition, args, done_func, reward_func)
    test_adv_dyna = AdversarialDynamics(all_state, offline_agent, transition, args, done_func, reward_func)
    return test_adv_dyna, offline_agent, adv_dyna

def policy_opt(offline_agent, adv_dyna):
    num_r = 0
    num_up = 0
    state = adv_dyna.reset()
    while num_up < args.agent_num_steps:
        if args.eval is True:
            evaluator.eval(num_up)
        for i_rollout in tqdm(range(1000), desc="{}th Thousand Steps".format(num_r)):
            action = offline_agent.act(state)
            _, adv_q, _, _ = adv_dyna.step(action)
            critic_loss, policy_loss = offline_agent.offline_update_parameters(state, action, adv_q)
            writer.add_scalar('loss_critic', critic_loss, num_up)
            writer.add_scalar('loss_policy', policy_loss, num_up)
            num_up += 1
            state = adv_dyna.state.cpu().numpy()
        num_r += 1

if __name__ == '__main__':
    args = args_setting()
    start_time = time.time()
    env = gym.make(args.task)
    env.seed(args.seed)
    setup_seed(args.seed)
    tadv, agent, adv = tran_model(env)
    data_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter('./log/policy/{}_{}_{}_{}'.format(
        data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))
    evaluator = Evaluator(env, tadv, agent, [1, 2, 3, 4], data_time, args)
    policy_opt(agent, adv)
    print("total time: {:.2f}s".format(time.time() - start_time))
    env.close()