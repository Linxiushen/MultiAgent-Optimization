from distributed.coordinator import DistributedCoordinator
import time

# 示例：展示如何使用分布式策略

def main():
    # 获取协调器实例
    coordinator = DistributedCoordinator()
    print("协调器初始化完成")

    # 注册多个Agent
    print("\n注册Agent...")
    agent_types = ['coding', 'research', 'planning', 'debugging']
    for i in range(5):
        agent_id = f'agent_{i}'
        agent_type = agent_types[i % len(agent_types)]
        capabilities = [agent_type, 'general']
        endpoint = f'http://localhost:800{i}'
        coordinator.register_agent(agent_id, agent_type, capabilities, endpoint)
        print(f"已注册Agent: {agent_id} (类型: {agent_type})")

    # 显示当前策略
    current_strategy = coordinator.config.get('load_balancing_strategy', 'round_robin')
    print(f"\n当前负载均衡策略: {current_strategy}")

    # 测试不同的策略
    strategies = ['round_robin', 'random', 'least_connections', 'performance_based', 'task_type_based']

    for strategy in strategies:
        print(f"\n切换到策略: {strategy}")
        if not coordinator.set_strategy(strategy):
            print(f"无效的策略: {strategy}，跳过")
            continue

        # 重置Agent任务计数
        for agent_id in coordinator.agents:
            coordinator.agents[agent_id]['task_count'] = 0
            # 为性能策略设置模拟性能分数
            coordinator.agents[agent_id]['performance_score'] = 100 - int(agent_id.split('_')[1]) * 10

        # 分配任务
        print("分配任务...")
        task_types = ['coding', 'research', 'planning', 'debugging', 'general']
        for i in range(10):
            task_type = task_types[i % len(task_types)]
            task_data = {'task_id': i, 'data': f'Task {i} data'}
            task_id = coordinator.assign_task(task_type, task_data)
            print(f"分配任务 {task_id} (类型: {task_type})")
            time.sleep(0.1)  # 短暂延迟，便于观察

        # 显示任务分配结果
        print("\n任务分配结果:")
        for agent_id, agent in coordinator.agents.items():
            print(f"Agent {agent_id} (类型: {agent['type']}): {agent['task_count']} 个任务")

        time.sleep(1)  # 暂停一下，便于观察

    print("\n示例程序执行完成")

if __name__ == '__main__':
    main()