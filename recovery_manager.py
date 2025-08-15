import time
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import json
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='recovery.log'
)
logger = logging.getLogger('RecoveryManager')

class RecoveryManager:
    def __init__(self, config_path: str = "configs/recovery_config.json"):
        """初始化故障恢复管理器

        Args:
            config_path: 恢复配置文件路径
        """
        self.config = self._load_config(config_path)
        self.failure_history: List[Dict[str, Any]] = []
        self.monitored_services: Dict[str, Dict[str, Any]] = {}
        self.recovery_callbacks: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
        self.is_running = False
        self.monitor_thread = None

        # 加载故障恢复配置
        self.failure_threshold = self.config.get("failure_threshold", 3)
        self.monitor_interval = self.config.get("monitor_interval", 60)  # 秒
        self.recovery_strategies = self.config.get("recovery_strategies", {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载恢复配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return {
                "failure_threshold": 3,
                "monitor_interval": 60,
                "recovery_strategies": {
                    "default": "restart_service"
                }
            }
        except json.JSONDecodeError:
            logger.error(f"配置文件 {config_path} 格式无效，使用默认配置")
            return {
                "failure_threshold": 3,
                "monitor_interval": 60,
                "recovery_strategies": {
                    "default": "restart_service"
                }
            }

    def register_service(self, service_id: str, health_check: Callable, recovery_strategy: str = "default"):
        """注册要监控的服务

        Args:
            service_id: 服务ID
            health_check: 健康检查函数，返回True表示健康，False表示故障
            recovery_strategy: 恢复策略名称
        """
        with self.lock:
            self.monitored_services[service_id] = {
                "health_check": health_check,
                "recovery_strategy": recovery_strategy,
                "failure_count": 0,
                "last_check_time": None,
                "status": "unknown"
            }
            self.recovery_callbacks[service_id] = []
            logger.info(f"服务 {service_id} 已注册，恢复策略: {recovery_strategy}")

    def unregister_service(self, service_id: str):
        """注销服务

        Args:
            service_id: 服务ID
        """
        with self.lock:
            if service_id in self.monitored_services:
                del self.monitored_services[service_id]
                logger.info(f"服务 {service_id} 已注销")
            if service_id in self.recovery_callbacks:
                del self.recovery_callbacks[service_id]

    def add_recovery_callback(self, service_id: str, callback: Callable):
        """添加恢复回调函数

        Args:
            service_id: 服务ID
            callback: 回调函数，恢复成功后调用
        """
        with self.lock:
            if service_id in self.recovery_callbacks:
                self.recovery_callbacks[service_id].append(callback)
                logger.info(f"为服务 {service_id} 添加恢复回调")
            else:
                logger.warning(f"服务 {service_id} 不存在，无法添加恢复回调")

    def start_monitoring(self):
        """开始监控服务"""
        if self.is_running:
            logger.warning("监控已经在运行中")
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("服务监控已启动")

    def stop_monitoring(self):
        """停止监控服务"""
        if not self.is_running:
            logger.warning("监控已经停止")
            return

        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("服务监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            self._check_all_services()
            time.sleep(self.monitor_interval)

    def _check_all_services(self):
        """检查所有服务的健康状态"""
        current_time = datetime.now().isoformat()

        with self.lock:
            for service_id, service_info in self.monitored_services.items():
                try:
                    # 执行健康检查
                    is_healthy = service_info["health_check"]()
                    service_info["last_check_time"] = current_time

                    if is_healthy:
                        # 服务健康
                        if service_info["status"] != "healthy":
                            service_info["status"] = "healthy"
                            service_info["failure_count"] = 0
                            logger.info(f"服务 {service_id} 恢复健康")
                    else:
                        # 服务故障
                        service_info["failure_count"] += 1
                        service_info["status"] = "unhealthy"
                        logger.warning(f"服务 {service_id} 健康检查失败，连续失败次数: {service_info['failure_count']}")

                        # 记录故障
                        self._record_failure(service_id, "health_check_failed")

                        # 达到故障阈值，执行恢复
                        if service_info["failure_count"] >= self.failure_threshold:
                            self._perform_recovery(service_id)
                except Exception as e:
                    # 健康检查本身失败
                    error_msg = f"服务 {service_id} 健康检查执行异常: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())

                    # 记录故障
                    self._record_failure(service_id, "health_check_exception", str(e))

                    # 增加故障计数
                    service_info["failure_count"] += 1
                    service_info["status"] = "error"

                    # 达到故障阈值，执行恢复
                    if service_info["failure_count"] >= self.failure_threshold:
                        self._perform_recovery(service_id)

    def _record_failure(self, service_id: str, failure_type: str, details: str = ""):
        """记录故障

        Args:
            service_id: 服务ID
            failure_type: 故障类型
            details: 故障详情
        """
        failure_record = {
            "service_id": service_id,
            "failure_type": failure_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.failure_history.append(failure_record)
        logger.error(f"记录故障: {service_id} - {failure_type} - {details}")

        # 保持故障历史不超过100条
        if len(self.failure_history) > 100:
            self.failure_history.pop(0)

    def _perform_recovery(self, service_id: str):
        """执行恢复操作

        Args:
            service_id: 服务ID
        """
        logger.info(f"开始执行服务 {service_id} 的恢复操作")

        try:
            service_info = self.monitored_services.get(service_id)
            if not service_info:
                logger.error(f"服务 {service_id} 不存在，无法执行恢复")
                return

            strategy_name = service_info["recovery_strategy"]
            strategy = self.recovery_strategies.get(strategy_name, "restart_service")

            # 执行恢复策略
            if strategy == "restart_service":
                # 重启服务策略 (示例实现)
                logger.info(f"执行重启服务策略 for {service_id}")
                # 这里应该包含实际的重启服务代码
                # 例如：调用服务的重启方法
                service_info["failure_count"] = 0
                service_info["status"] = "recovering"

                # 模拟重启延迟
                time.sleep(2)

                # 恢复后状态
                service_info["status"] = "healthy"
                logger.info(f"服务 {service_id} 已重启恢复")

            elif strategy == "switch_to_backup":
                # 切换到备份服务策略 (示例实现)
                logger.info(f"执行切换到备份服务策略 for {service_id}")
                # 这里应该包含实际的切换到备份服务代码
                service_info["failure_count"] = 0
                service_info["status"] = "healthy"
                logger.info(f"服务 {service_id} 已切换到备份服务")

            elif strategy == "notify_admin":
                # 通知管理员策略 (示例实现)
                logger.info(f"执行通知管理员策略 for {service_id}")
                # 这里应该包含实际的通知代码（如邮件、消息等）
                service_info["status"] = "unhealthy"
                logger.warning(f"已通知管理员服务 {service_id} 故障")

            else:
                # 默认策略
                logger.warning(f"未知的恢复策略: {strategy}，使用默认策略")
                service_info["failure_count"] = 0
                service_info["status"] = "recovering"
                time.sleep(2)
                service_info["status"] = "healthy"
                logger.info(f"服务 {service_id} 已使用默认策略恢复")

            # 执行恢复回调
            for callback in self.recovery_callbacks.get(service_id, []):
                try:
                    callback(service_id)
                except Exception as e:
                    logger.error(f"服务 {service_id} 恢复回调执行异常: {str(e)}")
                    logger.error(traceback.format_exc())

        except Exception as e:
            error_msg = f"服务 {service_id} 恢复操作失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self._record_failure(service_id, "recovery_failed", error_msg)

    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """获取服务状态

        Args:
            service_id: 服务ID

        Returns:
            服务状态信息，如果服务不存在则返回None
        """
        with self.lock:
            return self.monitored_services.get(service_id)

    def get_all_service_statuses(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务的状态

        Returns:
            所有服务的状态信息
        """
        with self.lock:
            return {k: v.copy() for k, v in self.monitored_services.items()}

    def get_failure_history(self, service_id: str = None) -> List[Dict[str, Any]]:
        """获取故障历史

        Args:
            service_id: 可选，服务ID，指定后只返回该服务的故障历史

        Returns:
            故障历史列表
        """
        with self.lock:
            if service_id:
                return [f for f in self.failure_history if f["service_id"] == service_id]
            else:
                return self.failure_history.copy()

# 单例模式
recovery_manager = RecoveryManager()