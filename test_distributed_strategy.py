import time
from distributed.coordinator import DistributedCoordinator
from distributed.distributed_strategy import StrategyFactory


def test_load_balancing_strategies():
    """测试不同的负载均衡策略"""
    # 获取协调器实例
    coordinator = DistributedCoordinator()

    # 注册多个Agent
    agent_types = ['coding', 'research', 'planning', 'debugging']
    for i in range(5):
        agent_id = f'agent_{i}'
        agent_type = agent_types[i % len(agent_types)]
        capabilities = [agent_type, 'general']
        endpoint = f'http://localhost:800{i}'
        coordinator.register_agent(agent_id, agent_type, capabilities, endpoint)

    # 测试不同的负载均衡策略
    strategies = ['round_robin', 'random', 'least_connections', 'performance_based', 'task_type_based']

    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        # 设置当前策略
        if not coordinator.set_strategy(strategy):
            print(f"跳过无效策略: {strategy}")
            continue

        # 重置Agent任务计数
        for agent_id in coordinator.agents:
            coordinator.agents[agent_id]['task_count'] = 0
            # 为性能基于策略设置模拟性能指标
            coordinator.agents[agent_id]['performance_score'] = 100 - i * 10 if strategy == 'performance_based' else 0

        # 分配任务
        task_types = ['coding', 'research', 'planning', 'debugging', 'general']
        for i in range(20):
            task_type = task_types[i % len(task_types)]
            task_data = {'task_id': i, 'data': f'Task {i} data'}
            coordinator.assign_task(task_type, task_data)

        # 打印任务分配结果
        print("任务分配结果:")
        for agent_id, agent in coordinator.agents.items():
            print(f"Agent {agent_id} (类型: {agent['type']}): {agent['task_count']} 个任务")

        # 短暂暂停，让任务分配完成
        time.sleep(1)


if __name__ == '__main__':
    print("开始测试分布式负载均衡策略...")
    test_load_balancing_strategies()
    print("测试完成！")