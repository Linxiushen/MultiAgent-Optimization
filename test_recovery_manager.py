import unittest
import time
from unittest.mock import Mock, patch
from recovery.recovery_manager import RecoveryManager

class TestRecoveryManager(unittest.TestCase):
    def setUp(self):
        # 获取恢复管理器单例实例
        self.recovery_manager = RecoveryManager()
        # 重置恢复管理器状态
        self.recovery_manager.services = {}
        self.recovery_manager.failure_records = {}
        self.recovery_manager.config = {
            "failure_threshold": 3,
            "monitor_interval": 60,
            "recovery_strategies": {
                "default": "restart_service",
                "llm_service": "switch_to_backup",
                "memory_service": "restart_service",
                "router_service": "notify_admin",
                "critical_service": "auto_restart_and_notify"
            },
            "backup_services": {
                "llm_service": {
                    "primary": "localhost:8001",
                    "backup": "localhost:8002"
                }
            },
            "admin_notifications": {
                "email": "admin@example.com",
                "slack_channel": "#system-alerts"
            },
            "auto_restart_delay": 5
        }

    def test_singleton_pattern(self):
        # 测试单例模式
        another_manager = RecoveryManager()
        self.assertIs(self.recovery_manager, another_manager, "RecoveryManager不是单例模式")

    def test_register_service(self):
        # 测试服务注册
        service_id = 'llm_service'
        result = self.recovery_manager.register_service(
            service_id=service_id,
            service_type='llm',
            health_check_callback=lambda: True,
            recovery_callback=lambda: True
        )
        self.assertTrue(result, "服务注册失败")
        self.assertIn(service_id, self.recovery_manager.services, "注册的服务不在服务列表中")
        self.assertEqual(self.recovery_manager.services[service_id]['type'], 'llm', "服务类型不正确")
        self.assertEqual(self.recovery_manager.services[service_id]['status'], 'healthy', "服务初始状态应为healthy")

    def test_unregister_service(self):
        # 测试服务注销
        service_id = 'memory_service'
        self.recovery_manager.register_service(
            service_id=service_id,
            service_type='memory',
            health_check_callback=lambda: True,
            recovery_callback=lambda: True
        )
        result = self.recovery_manager.unregister_service(service_id)
        self.assertTrue(result, "服务注销失败")
        self.assertNotIn(service_id, self.recovery_manager.services, "注销的服务仍在服务列表中")

    def test_health_check(self):
        # 测试健康检查
        # 健康服务
        healthy_service_id = 'healthy_service'
        self.recovery_manager.register_service(
            service_id=healthy_service_id,
            service_type='test',
            health_check_callback=lambda: True,
            recovery_callback=lambda: True
        )
        healthy_result = self.recovery_manager.check_service_health(healthy_service_id)
        self.assertTrue(healthy_result, "健康检查结果不正确")
        self.assertEqual(self.recovery_manager.services[healthy_service_id]['status'], 'healthy', "健康服务状态不正确")

        # 不健康服务
        unhealthy_service_id = 'unhealthy_service'
        self.recovery_manager.register_service(
            service_id=unhealthy_service_id,
            service_type='test',
            health_check_callback=lambda: False,
            recovery_callback=lambda: True
        )
        unhealthy_result = self.recovery_manager.check_service_health(unhealthy_service_id)
        self.assertFalse(unhealthy_result, "健康检查结果不正确")
        self.assertEqual(self.recovery_manager.services[unhealthy_service_id]['status'], 'unhealthy', "不健康服务状态不正确")

    def test_detect_failure(self):
        # 测试故障检测
        service_id = 'failure_service'
        self.recovery_manager.register_service(
            service_id=service_id,
            service_type='test',
            health_check_callback=lambda: False,
            recovery_callback=lambda: True
        )

        # 连续检查多次，触发故障检测
        for _ in range(4):  # 超过故障阈值3次
            self.recovery_manager.check_service_health(service_id)

        self.assertIn(service_id, self.recovery_manager.failure_records, "故障未记录")
        self.assertEqual(self.recovery_manager.failure_records[service_id]['failure_count'], 4, "故障计数不正确")
        self.assertTrue(self.recovery_manager.failure_records[service_id]['is_critical'], "故障严重程度标记不正确")

    @patch('recovery.recovery_manager.RecoveryManager.restart_service')
    @patch('recovery.recovery_manager.RecoveryManager.switch_to_backup')
    @patch('recovery.recovery_manager.RecoveryManager.notify_admin')
    def test_recovery_strategies(self, mock_notify_admin, mock_switch_to_backup, mock_restart_service):
        # 测试不同的恢复策略
        # 默认策略 (restart_service)
        default_service_id = 'default_service'
        self.recovery_manager.register_service(
            service_id=default_service_id,
            service_type='default',
            health_check_callback=lambda: False,
            recovery_callback=lambda: True
        )
        self.recovery_manager._handle_recovery(default_service_id)
        mock_restart_service.assert_called_once_with(default_service_id)

        # llm_service策略 (switch_to_backup)
        llm_service_id = 'llm_service'
        self.recovery_manager.register_service(
            service_id=llm_service_id,
            service_type='llm_service',
            health_check_callback=lambda: False,
            recovery_callback=lambda: True
        )
        self.recovery_manager._handle_recovery(llm_service_id)
        mock_switch_to_backup.assert_called_once_with(llm_service_id)

        # router_service策略 (notify_admin)
        router_service_id = 'router_service'
        self.recovery_manager.register_service(
            service_id=router_service_id,
            service_type='router_service',
            health_check_callback=lambda: False,
            recovery_callback=lambda: True
        )
        self.recovery_manager._handle_recovery(router_service_id)
        mock_notify_admin.assert_called_once_with(router_service_id)

    def test_get_service_status(self):
        # 测试获取服务状态
        service_id = 'status_service'
        self.recovery_manager.register_service(
            service_id=service_id,
            service_type='test',
            health_check_callback=lambda: True,
            recovery_callback=lambda: True
        )

        status = self.recovery_manager.get_service_status(service_id)
        self.assertEqual(status['service_id'], service_id, "服务ID不正确")
        self.assertEqual(status['type'], 'test', "服务类型不正确")
        self.assertEqual(status['status'], 'healthy', "服务状态不正确")
        self.assertEqual(status['failure_count'], 0, "故障计数不正确")

    def test_get_all_services_status(self):
        # 测试获取所有服务状态
        self.recovery_manager.register_service(
            service_id='service1',
            service_type='test1',
            health_check_callback=lambda: True,
            recovery_callback=lambda: True
        )
        self.recovery_manager.register_service(
            service_id='service2',
            service_type='test2',
            health_check_callback=lambda: False,
            recovery_callback=lambda: True
        )

        statuses = self.recovery_manager.get_all_services_status()
        self.assertEqual(len(statuses), 2, "返回的服务状态数量不正确")
        self.assertIn('service1', statuses, "service1状态未返回")
        self.assertIn('service2', statuses, "service2状态未返回")
        self.assertEqual(statuses['service1']['status'], 'healthy', "service1状态不正确")
        self.assertEqual(statuses['service2']['status'], 'unhealthy', "service2状态不正确")

if __name__ == '__main__':
    unittest.main()