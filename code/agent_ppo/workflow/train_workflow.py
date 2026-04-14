#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
清扫大作战训练工作流。
"""

import os
import random
import time

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate user configuration
    # 读取和校验用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    # Keep a fraction of guard-forced samples to avoid starving PPO updates.
    GUARD_SAMPLE_KEEP_PROB = 0.35

    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        单局流程（generator），完成一局后 yield 整局样本。
        """
        while True:
            # Periodically get training metrics
            # 定期打印训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            # Reset environment
            # 重置环境
            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent and load latest model
            # 重置 Agent，加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation processing
            # 初始观测
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            skipped_guard_steps = 0
            kept_guard_steps = 0
            last_env_obs = env_obs

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Agent inference / 推理动作
                act_data_list = self.agent.predict([obs_data])
                act_data = act_data_list[0]
                act = self.agent.action_process(act_data)

                # Environment step / 与环境交互
                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated
                last_env_obs = env_obs

                # Process next observation
                # 特征处理
                _obs_data, _ = self.agent.observation_process(env_obs, step_env_reward=env_reward)
                _obs_data.frame_no = frame_no

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                # Terminal reward calculation
                # 终局奖励
                final_reward = 0.0
                if done:
                    fm = self.agent.preprocessor
                    cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                    env_info = ((last_env_obs or {}).get("observation") or {}).get("env_info") or {}
                    extra_info = (last_env_obs or {}).get("extra_info") or {}
                    charge_count = int(env_info.get("charge_count", 0))
                    remaining_charge = int(env_info.get("remaining_charge", fm.battery))
                    nearest_charger_dist = float(getattr(fm, "nearest_charger_dist", 200.0))
                    battery_minus_charger_dist = float(remaining_charge) - float(nearest_charger_dist)
                    guard_count = int(getattr(fm, "charge_guard_trigger_count", 0))
                    guard_min_charger_dist = float(getattr(fm, "guard_min_charger_dist", 200.0))
                    guard_terminal_override_count = int(getattr(fm, "guard_terminal_override_count", 0))
                    result_code = int(extra_info.get("result_code", -1))
                    result_message = str(extra_info.get("result_message", ""))

                    if truncated:
                        # Survived to max steps: align bonus with official "more tiles cleaned"
                        # 存活满步：清扫比例越高终局奖励越大（与得分方向一致）
                        final_reward = 6.0 + 8.0 * cleaning_ratio
                        result_str = "WIN"
                    else:
                        # Early end: penalize less when already cleaned many tiles (same as score emphasis)
                        # 提前结束：清扫比例高则减轻惩罚，避免策略只赌「苟活」而忽视扫地
                        final_reward = -2.0 + 5.0 * cleaning_ratio
                        result_str = "FAIL"

                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} final_bonus:{final_reward:.2f} "
                        f"total_reward:{total_reward:.3f} "
                        f"dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt} "
                        f"charge_count:{charge_count} remain:{remaining_charge} "
                        f"nearest_charger_dist:{nearest_charger_dist:.2f} "
                        f"battery_minus_charger_dist:{battery_minus_charger_dist:.2f} "
                        f"guard_count:{guard_count} "
                        f"skipped_guard_steps:{skipped_guard_steps} "
                        f"kept_guard_steps:{kept_guard_steps} "
                        f"guard_min_charger_dist:{guard_min_charger_dist:.2f} "
                        f"guard_terminal_override_count:{guard_terminal_override_count} "
                        f"terminated:{terminated} truncated:{truncated} "
                        f"result_code:{result_code} result_message:{result_message}"
                    )

                # Build sample frame
                # 构造样本帧
                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action),
                    reward=reward_arr,
                    done=np.array([float(done)]),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                # Skip safety-override transitions to reduce off-policy noise in PPO updates.
                if int(getattr(self.agent.preprocessor, "charge_guard_triggered", 0)) == 1:
                    if random.random() < self.GUARD_SAMPLE_KEEP_PROB:
                        collector.append(frame)
                        kept_guard_steps += 1
                    else:
                        skipped_guard_steps += 1
                else:
                    collector.append(frame)

                if done:
                    # Add terminal reward to last frame
                    # 终局奖励叠加到最后一步
                    if collector:
                        collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    # Monitor reporting / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "reward": total_reward + final_reward,
                                    "episode_cnt": self.episode_cnt,
                                    "charge_count": charge_count,
                                    "remaining_charge": remaining_charge,
                                    "nearest_charger_dist": nearest_charger_dist,
                                    "battery_minus_charger_dist": battery_minus_charger_dist,
                                    "charge_guard_count": guard_count,
                                    "skipped_guard_steps": skipped_guard_steps,
                                    "kept_guard_steps": kept_guard_steps,
                                    "guard_min_charger_dist": guard_min_charger_dist,
                                    "guard_terminal_override_count": guard_terminal_override_count,
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    # Compute GAE and yield samples
                    # GAE 计算并 yield 样本
                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Advance state / 状态推进
                obs_data = _obs_data
