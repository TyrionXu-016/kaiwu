#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Robot Vacuum.
清扫大作战监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫大作战")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="每局充电次数",
            name_en="charge_count",
            type="line",
        )
        .add_metric(
            metrics_name="charge_count",
            expr="avg(charge_count{})",
        )
        .end_panel()
        .add_panel(
            name="结束剩余电量",
            name_en="remaining_charge",
            type="line",
        )
        .add_metric(
            metrics_name="remaining_charge",
            expr="avg(remaining_charge{})",
        )
        .end_panel()
        .add_panel(
            name="最近充电桩距离",
            name_en="nearest_charger_dist",
            type="line",
        )
        .add_metric(
            metrics_name="nearest_charger_dist",
            expr="avg(nearest_charger_dist{})",
        )
        .end_panel()
        .add_panel(
            name="电量-充电桩距离裕量",
            name_en="battery_minus_charger_dist",
            type="line",
        )
        .add_metric(
            metrics_name="battery_minus_charger_dist",
            expr="avg(battery_minus_charger_dist{})",
        )
        .end_panel()
        .add_panel(
            name="硬回充触发次数",
            name_en="charge_guard_count",
            type="line",
        )
        .add_metric(
            metrics_name="charge_guard_count",
            expr="avg(charge_guard_count{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
