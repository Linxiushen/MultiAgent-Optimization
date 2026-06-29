import sys
import os
import time
import threading
import random
from queue import PriorityQueue

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed.coordinator import DistributedCoordinator

def simulate_agent_heartbeat(coordinator, agent_id, failure_time=None):
    """模拟Agent发送心跳的线程函数
    
    Args:
        coordinator: 分布式协调器实例
        agent_id: Agent ID
        failure_time: 如果设置，在指定时间后停止发送心跳，模拟故障
    """
    start_time = time.time()
    while True:
        # 如果设置了故障时间，并且已经达到，则停止发送心跳
        if failure_time and time.time() - start_time > failure_time:
            print(f"Agent {agent_id} 模拟故障，停止发送心跳")
            break
            
        # 发送心跳
        coordinator.update_agent_heartbeat(agent_id)
        print(f"Agent {agent_id} 发送心跳")
        
        # 随机等待一段时间
        time.sleep(random.uniform(1, 3))

def simulate_task_execution(coordinator, agent_id):
    """模拟Agent执行任务的线程函数
    
    Args:
        coordinator: 分布式协调器实例
        agent_id: Agent ID
    """
    while True:
        # 获取分配给该Agent的任务
        agent_tasks = []
        temp_queue = PriorityQueue()
        
        while not coordinator.tasks.empty():
            priority, timestamp, task = coordinator.tasks.get()
            if task['assigned_to'] == agent_id and 'completed' not in task:
                agent_tasks.append((priority, timestamp, task))
            else:
                temp_queue.put((priority, timestamp, task))
        
        # 恢复其他任务到队列
        while not temp_queue.empty():
            coordinator.tasks.put(temp_queue.get())
        
        # 执行任务
        for priority, timestamp, task in agent_tasks:
            # 模拟任务执行
            print(f"Agent {agent_id} 执行任务 {task['id']}")
            execution_time = random.uniform(2, 5)
            time.sleep(execution_time)
            
            # 标记任务完成
            task['completed'] = True
            task['completion_time'] = time.time()
            
            print(f"Agent {agent_id} 完成任务 {task['id']}，耗时 {execution_time:.2f} 秒")
            
            # 更新Agent任务计数
            coordinator.agents[agent_id]['task_count'] -= 1
        
        # 如果没有任务，等待一段时间
        if not agent_tasks:
            time.sleep(1)

def main():
    # 创建协调器实例
    coordinator = DistributedCoordinator()
    
    # 注册Agent
    coordinator.register_agent('agent1', 'worker', ['task_type1', 'task_type2'], 'http://agent1')
    coordinator.register_agent('agent2', 'worker', ['task_type1', 'task_type3'], 'http://agent2')
    coordinator.register_agent('agent3', 'worker', ['task_type2', 'task_type3'], 'http://agent3')
    
    # 创建任务
    tasks = [
        {
            'id': 'task1',
            'type': 'task_type1',
            'priority': 5,
            'data': {'input': 'data1'}
        },
        {
            'id': 'task2',
            'type': 'task_type2',
            'priority': 3,
            'data': {'input': 'data2'}
        },
        {
            'id': 'task3',
            'type': 'task_type3',
            'priority': 7,
            'data': {'input': 'data3'}
        },
        {
            'id': 'task4',
            'type': 'task_type1',
            'priority': 2,
            'data': {'input': 'data4'}
        },
        {
            'id': 'task5',
            'type': 'task_type2',
            'priority': 6,
            'data': {'input': 'data5'}
        }
    ]
    
    # 分配任务
    for task in tasks:
        assigned_agent = coordinator.assign_task(task)
        print(f"任务 {task['id']} 分配给 Agent {assigned_agent}")
    
    # 启动Agent心跳线程
    heartbeat_threads = [
        threading.Thread(target=simulate_agent_heartbeat, args=(coordinator, 'agent1', 15)),  # agent1将在15秒后故障
        threading.Thread(target=simulate_agent_heartbeat, args=(coordinator, 'agent2')),
        threading.Thread(target=simulate_agent_heartbeat, args=(coordinator, 'agent3'))
    ]
    
    # 启动Agent任务执行线程
    execution_threads = [
        threading.Thread(target=simulate_task_execution, args=(coordinator, 'agent1')),
        threading.Thread(target=simulate_task_execution, args=(coordinator, 'agent2')),
        threading.Thread(target=simulate_task_execution, args=(coordinator, 'agent3'))
    ]
    
    # 设置为守护线程并启动
    for thread in heartbeat_threads + execution_threads:
        thread.daemon = True
        thread.start()
    
    # 运行演示30秒
    try:
        print("故障转移演示开始，将运行30秒...")
        time.sleep(30)
        
        # 打印集群状态
        status = coordinator.get_cluster_status()
        print("\n集群状态:")
        print(f"活跃Agent: {status['active_agents']}")
        print(f"不活跃Agent: {status['inactive_agents']}")
        print(f"待处理任务: {status['pending_tasks']}")
        print(f"故障事件: {len(coordinator.failure_history)}")
        
        # 打印故障历史
        print("\n故障历史:")
        for event in coordinator.failure_history:
            print(f"时间: {event['timestamp']}, Agent: {event['agent_id']}, 类型: {event['failure_type']}")
        
    except KeyboardInterrupt:
        print("演示被用户中断")
    
    print("故障转移演示结束")

if __name__ == "__main__":
    main()