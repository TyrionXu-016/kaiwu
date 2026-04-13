#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np


def _coerce_env_reward(x):
    if x is None:
        return 0.0
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return 0.0
            return float(np.asarray(x).reshape(-1)[0])
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径

    APPROACH_DIRT_REWARD = 0.004
    CHARGE_GAIN_COEF = 0.006
    CHARGER_CELL_VALUES = (3, 4)
    APPROACH_CHARGER_REWARD = 0.008
    LOW_BATTERY_RATIO = 0.42
    CLEANING_TILE_WEIGHT = 0.18
    STEP_PENALTY_IDLE = -0.0012
    STEP_PENALTY_ACTIVE = -0.0004
    ENV_REWARD_BLEND = 0.12

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self.nearest_charger_dist = 200.0
        self.last_nearest_charger_dist = 200.0

        self._prev_battery = 0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8
        self._frame_env_reward = 0.0

    def pb2struct(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        prev_battery = self.battery

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)
        self._prev_battery = prev_battery

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

    def _update_passable(self, hx, hz):
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = self._calc_nearest_charger_dist()
        nearest_charger_norm = _norm(self.nearest_charger_dist, 180.0)
        charger_delta = 0.0
        if battery_ratio < self.LOW_BATTERY_RATIO and self.nearest_charger_dist < self.last_nearest_charger_dist:
            charger_delta = 1.0

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
                nearest_charger_norm,
                charger_delta,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _calc_nearest_charger_dist(self):
        view = self._view_map
        if view is None:
            return 200.0
        mask = np.zeros_like(view, dtype=bool)
        for v in self.CHARGER_CELL_VALUES:
            mask |= view == float(v)
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((coords[:, 0] - center) ** 2 + (coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action, step_env_reward=0.0):
        self.pb2struct(env_obs, last_action)
        self._frame_env_reward = _coerce_env_reward(step_env_reward)

        local_view = self._get_local_view_feature()
        global_state = self._get_global_state_feature()
        legal_action = self.get_legal_action()
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = self.CLEANING_TILE_WEIGHT * float(cleaned_this_step)
        step_penalty = (
            self.STEP_PENALTY_ACTIVE if cleaned_this_step > 0 else self.STEP_PENALTY_IDLE
        )
        approach_reward = (
            self.APPROACH_DIRT_REWARD if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        )
        gained = max(0, self.battery - self._prev_battery)
        charge_reward = self.CHARGE_GAIN_COEF * min(gained, 80)
        br = float(self.battery) / float(max(self.battery_max, 1))
        approach_charger = 0.0
        if br < self.LOW_BATTERY_RATIO and self.nearest_charger_dist < self.last_nearest_charger_dist:
            approach_charger = self.APPROACH_CHARGER_REWARD
        env_r = self.ENV_REWARD_BLEND * self._frame_env_reward
        return cleaning_reward + step_penalty + approach_reward + charge_reward + approach_charger + env_r
