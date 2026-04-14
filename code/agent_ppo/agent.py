#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent.
清扫大作战 Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs, step_env_reward=0.0):
        """Convert raw env_obs to ObsData (71D feature + legal action mask).

        step_env_reward: pass env.step's first return to align RL reward with official score signal.
        """
        feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, self.last_action, step_env_reward=step_env_reward
        )
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 中取出动作整数并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时推理（随机采样动作）。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)

        legal_arr = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_arr)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        # Keep the behavior policy consistent with the executed action.
        # If we override action by rules, record one-hot probs for PPO ratio correctness.
        behavior_prob = prob.copy()
        action_overridden = False

        # Low-battery hard guard: force heading to nearest charger when needed.
        # 低电量硬保护：必要时强制朝最近充电桩移动。
        guard_action = self.preprocessor.get_charge_guard_action(legal_action, last_action=self.last_action)
        if guard_action is not None:
            action = guard_action
            d_action = guard_action
            action_overridden = True
        else:
            # In normal mode, avoid one-step NPC danger cells when possible.
            # 非回充模式下优先过滤一步 NPC 危险落脚点，降低高电量早停风险。
            safe_legal = self.preprocessor.get_npc_safe_action_mask(legal_action, danger_radius=1)
            safe_arr = np.array(safe_legal, dtype=np.float32)
            prob = self._legal_soft_max(logits, safe_arr)
            action = self._legal_sample(prob, use_max=False)
            d_action = self._legal_sample(prob, use_max=True)

            # Coverage planner first: select frontier target and move toward it.
            # 覆盖规划优先：朝 frontier 目标推进。
            frontier_action = self.preprocessor.get_frontier_action(
                legal_action=safe_legal, last_action=self.last_action
            )
            if frontier_action is not None:
                action = frontier_action
                d_action = frontier_action
                action_overridden = True
            else:
                # Fallback to mild cardinal bias.
                # 兜底使用轻量直行动作偏置。
                cardinal_action = self.preprocessor.get_cardinal_clean_action(
                    legal_action=safe_legal, probs=prob, last_action=self.last_action
                )
                if cardinal_action is not None:
                    d_action = cardinal_action

        if action_overridden:
            behavior_prob = np.zeros_like(prob, dtype=np.float32)
            behavior_prob[int(action)] = 1.0

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(behavior_prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时推理（贪心）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行训练。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _run_model(self, feature):
        """Gradient-free forward pass, returns (logits_np, value_np).

        无梯度推理，返回 (logits_np, value_np)。
        """
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.array([feature], dtype=np.float32)).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)
        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _legal_soft_max(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True).

        按概率分布采样动作（use_max=True 时取 argmax）。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
