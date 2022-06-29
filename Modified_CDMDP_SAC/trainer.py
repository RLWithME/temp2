from collections import deque

import numpy as np
from utils import log_to_txt


class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args

        self.agent = agent
        self.delayed_env = env
        self.eval_delayed_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        # Delay 관련 변수
        self.delay_step = args.delay_step
        self.action_buffer = deque(maxlen=args.delay_step)
        self.eval_action_buffer = deque(maxlen=args.delay_step)

        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.done_mask_history = []
        self.obs_history = []

    def init_eval_buffer(self):
        self.eval_action_buffer.clear()
        for _ in range(self.args.delay_step):
            self.eval_action_buffer.append(np.zeros_like(self.delayed_env.action_space.sample()))

    def init_history_and_buffer(self):
        self.state_history = []
        self.action_history = []
        self.next_state_history = []
        self.reward_history = []
        self.done_mask_history = []
        self.obs_history = []
        self.action_buffer.clear()
        for _ in range(self.args.delay_step):
            self.action_buffer.append(np.zeros_like(self.delayed_env.action_space.sample()))

    def get_true_transition(self, state_buffer, action_buffer, next_state_buffer, reward_buffer, done_mask_buffer, obs_buffer, delay_step):
        assert len(state_buffer) == len(obs_buffer)
        assert delay_step > 0
        last_index = len(state_buffer) - 1

        state = state_buffer[last_index - delay_step]
        action = action_buffer[last_index - delay_step]
        next_state = next_state_buffer[last_index - delay_step]
        reward = reward_buffer[last_index]
        done_mask = done_mask_buffer[last_index]
        obs = obs_buffer[last_index - 1]
        next_obs = obs_buffer[last_index]
        return state, action, reward, next_state, done_mask, obs, next_obs

    def get_state(self, last_obs, actions):
        state = np.concatenate([last_obs, actions[0]])
        for i in range(1, self.delay_step):
            state = np.concatenate([state, actions[i]])
        return state

    def evaluate(self):
        # Evaluate process
        self.eval_num += 1
        reward_list = []

        for epi in range(self.eval_episode):
            epi_reward = 0
            # initialize delay history and buffer
            self.init_eval_buffer()
            last_obs = self.eval_delayed_env.reset()
            state = self.get_state(last_obs, self.eval_action_buffer)

            done = False

            while not done:
                action = self.agent.get_action(state, evaluation=True)
                self.eval_action_buffer.popleft()
                self.eval_action_buffer.append(action)
                next_last_obs, reward, done, _ = self.eval_delayed_env.step(action)
                epi_reward += reward
                next_state = self.get_state(next_last_obs, self.eval_action_buffer)
                state = next_state
            reward_list.append(epi_reward)

        if self.args.log:
            log_to_txt(self.args, self.total_step, sum(reward_list)/len(reward_list))
        print("Eval  |  total_step {}  |  episode {}  |  Average Reward {:.2f}  |  Max reward: {:.2f}  |  "
              "Min reward: {:.2f}".format(self.total_step, self.episode, sum(reward_list)/len(reward_list),
                                               max(reward_list), min(reward_list), np.std(reward_list)))

    def run(self):
        # Train-process start.
        while not self.finish_flag:
            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            # 에피소드를 실핼하기 전 가장 먼저 모든 히스토리와 버퍼를 초기화한다.
            self.init_history_and_buffer()
            last_obs = self.delayed_env.reset()
            state = self.get_state(last_obs, self.action_buffer)
            done = False

            # Episode start.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.total_step >= self.start_step:
                    action = self.agent.get_action(state, evaluation=False)
                    self.action_buffer.popleft()
                    self.action_buffer.append(action)
                else:
                    # 충분히 샘플을 모을 때 까지 랜덤으로 행동을 정한다.
                    action = self.delayed_env.action_space.sample()
                    self.action_buffer.popleft()
                    self.action_buffer.append(action)

                # 참고: 행동 이후 관측된 상태, 보상, 종료 여부(next_obs, reward, done)는 지연된 상태이다.
                next_last_obs, reward, done, _ = self.delayed_env.step(action)
                next_state = self.get_state(next_last_obs, self.action_buffer)
                self.episode_reward += reward

                done_mask = 0.0 if self.local_step == self.delayed_env._max_episode_steps + self.args.delay_step else float(done)

                self.state_history.append(state)
                self.next_state_history.append(next_state)
                self.action_history.append(action)
                self.reward_history.append(reward)
                self.done_mask_history.append(done_mask)
                self.obs_history.append(next_last_obs)

                # 히스토리 버퍼에서 "참 transition 튜플"을 얻는다.
                if len(self.state_history) > self.delay_step:
                    _state, _action, _reward, _next_state, _done_mask, _, _ = self.get_true_transition(self.state_history, self.action_history, self.next_state_history, self.reward_history, self.done_mask_history,  self.obs_history, self.delay_step)
                    self.agent.buffer.push(_state, _action, _reward, _next_state, _done_mask)

                state = next_state

                # Update parameters
                if self.agent.buffer.size >= self.batch_size and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:
                    total_actor_loss = 0
                    total_critic_loss = 0
                    total_log_alpha_loss = 0
                    for i in range(self.update_every):
                        critic_loss, actor_loss, log_alpha_loss = self.agent.train()
                        total_critic_loss += critic_loss
                        total_actor_loss += actor_loss
                        total_log_alpha_loss += log_alpha_loss

                    # Print loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}  |  Log-alpha loss {:.3f}"
                              .format(total_actor_loss / self.update_every, total_critic_loss / self.update_every,
                                      total_log_alpha_loss / self.update_every))

                # Evaluation.
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish_flag.
                if self.total_step == self.max_step:
                    self.finish_flag = True










