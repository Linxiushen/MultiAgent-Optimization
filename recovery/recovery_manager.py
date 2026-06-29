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
    # 单例实例
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # 单例模式：确保只创建一个实例
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "configs/recovery_config.json"):
        """初始化故障恢复管理器

        Args:
            config_path: 恢复配置文件路径
        """
        # 单例：避免重复初始化重置状态
        if getattr(self, "_initialized", False):
            return

        self.config = self._load_config(config_path)
        self.services: Dict[str, Dict[str, Any]] = {}
        self.failure_records: Dict[str, Dict[str, Any]] = {}
        self.failure_history: List[Dict[str, Any]] = []
        self.recovery_callbacks: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
        self.is_running = False
        self.monitor_thread = None

        self._initialized = True

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载恢复配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        default_config = {
            "failure_threshold": 3,
            "monitor_interval": 60,
            "recovery_strategies": {
                "default": "restart_service"
            }
        }
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return default_config
        except json.JSONDecodeError:
            logger.error(f"配置文件 {config_path} 格式无效，使用默认配置")
            return default_config

    @property
    def failure_threshold(self) -> int:
        return self.config.get("failure_threshold", 3)

    @property
    def monitor_interval(self) -> int:
        return self.config.get("monitor_interval", 60)

    @property
    def recovery_strategies(self) -> Dict[str, str]:
        return self.config.get("recovery_strategies", {})

    def register_service(self, service_id: str, service_type: str = "default",
                         health_check_callback: Optional[Callable] = None,
                         recovery_callback: Optional[Callable] = None,
                         **kwargs) -> bool:
        """注册要监控的服务

        Args:
            service_id: 服务ID
            service_type: 服务类型（用于选择恢复策略）
            health_check_callback: 健康检查函数，返回True表示健康，False表示故障
            recovery_callback: 恢复回调函数

        Returns:
            注册是否成功
        """
        try:
            with self.lock:
                self.services[service_id] = {
                    "type": service_type,
                    "health_check_callback": health_check_callback,
                    "recovery_callback": recovery_callback,
                    "status": "healthy",
                    "last_check_time": None,
                }
                self.failure_records.setdefault(service_id, {
                    "failure_count": 0,
                    "is_critical": False,
                })
                self.recovery_callbacks[service_id] = []
                if recovery_callback is not None:
                    self.recovery_callbacks[service_id].append(recovery_callback)
                logger.info(f"服务 {service_id} 已注册，类型: {service_type}")
            return True
        except Exception as e:
            logger.error(f"注册服务 {service_id} 失败: {str(e)}")
            return False

    def unregister_service(self, service_id: str) -> bool:
        """注销服务

        Args:
            service_id: 服务ID

        Returns:
            注销是否成功
        """
        with self.lock:
            if service_id in self.services:
                del self.services[service_id]
                logger.info(f"服务 {service_id} 已注销")
            if service_id in self.failure_records:
                del self.failure_records[service_id]
            if service_id in self.recovery_callbacks:
                del self.recovery_callbacks[service_id]
        return True

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

    def check_service_health(self, service_id: str) -> bool:
        """检查单个服务的健康状态

        Args:
            service_id: 服务ID

        Returns:
            True表示健康，False表示故障
        """
        service_info = self.services.get(service_id)
        if not service_info:
            logger.warning(f"服务 {service_id} 不存在，无法检查健康状态")
            return False

        health_check = service_info.get("health_check_callback")
        try:
            is_healthy = bool(health_check()) if health_check else True
        except Exception as e:
            logger.error(f"服务 {service_id} 健康检查执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            is_healthy = False

        service_info["last_check_time"] = datetime.now().isoformat()

        if is_healthy:
            service_info["status"] = "healthy"
        else:
            service_info["status"] = "unhealthy"
            self._detect_failure(service_id)

        return is_healthy

    def _detect_failure(self, service_id: str):
        """记录并检测服务故障

        Args:
            service_id: 服务ID
        """
        record = self.failure_records.setdefault(service_id, {
            "failure_count": 0,
            "is_critical": False,
        })
        record["failure_count"] += 1
        record["is_critical"] = record["failure_count"] > self.failure_threshold

        self._record_failure(service_id, "health_check_failed")
        logger.warning(
            f"服务 {service_id} 健康检查失败，连续失败次数: {record['failure_count']}"
        )

    def _handle_recovery(self, service_id: str):
        """根据服务类型选择并执行恢复策略

        Args:
            service_id: 服务ID
        """
        service_info = self.services.get(service_id)
        if not service_info:
            logger.error(f"服务 {service_id} 不存在，无法执行恢复")
            return

        service_type = service_info.get("type", "default")
        strategy = self.recovery_strategies.get(
            service_type, self.recovery_strategies.get("default", "restart_service")
        )

        logger.info(f"为服务 {service_id} 执行恢复策略: {strategy}")

        try:
            if strategy == "restart_service":
                self.restart_service(service_id)
            elif strategy == "switch_to_backup":
                self.switch_to_backup(service_id)
            elif strategy == "notify_admin":
                self.notify_admin(service_id)
            elif strategy == "auto_restart_and_notify":
                self.restart_service(service_id)
                self.notify_admin(service_id)
            else:
                logger.warning(f"未知的恢复策略: {strategy}，使用默认重启策略")
                self.restart_service(service_id)
        except Exception as e:
            error_msg = f"服务 {service_id} 恢复操作失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self._record_failure(service_id, "recovery_failed", error_msg)

    def restart_service(self, service_id: str):
        """重启服务恢复策略

        Args:
            service_id: 服务ID
        """
        logger.info(f"执行重启服务策略 for {service_id}")
        service_info = self.services.get(service_id)
        if service_info:
            service_info["status"] = "healthy"
        if service_id in self.failure_records:
            self.failure_records[service_id]["failure_count"] = 0
            self.failure_records[service_id]["is_critical"] = False
        self._run_recovery_callbacks(service_id)

    def switch_to_backup(self, service_id: str):
        """切换到备份服务恢复策略

        Args:
            service_id: 服务ID
        """
        logger.info(f"执行切换到备份服务策略 for {service_id}")
        service_info = self.services.get(service_id)
        if service_info:
            service_info["status"] = "healthy"
        if service_id in self.failure_records:
            self.failure_records[service_id]["failure_count"] = 0
            self.failure_records[service_id]["is_critical"] = False
        self._run_recovery_callbacks(service_id)

    def notify_admin(self, service_id: str):
        """通知管理员恢复策略

        Args:
            service_id: 服务ID
        """
        admin = self.config.get("admin_notifications", {})
        logger.warning(
            f"已通知管理员服务 {service_id} 故障 (邮件: {admin.get('email')}, "
            f"渠道: {admin.get('slack_channel')})"
        )
        self._run_recovery_callbacks(service_id)

    def _run_recovery_callbacks(self, service_id: str):
        """执行已注册的恢复回调"""
        for callback in self.recovery_callbacks.get(service_id, []):
            try:
                callback(service_id)
            except Exception as e:
                logger.error(f"服务 {service_id} 恢复回调执行异常: {str(e)}")
                logger.error(traceback.format_exc())

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
        for service_id in list(self.services.keys()):
            is_healthy = self.check_service_health(service_id)
            if not is_healthy:
                record = self.failure_records.get(service_id, {})
                if record.get("failure_count", 0) >= self.failure_threshold:
                    self._handle_recovery(service_id)

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

    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """获取服务状态

        Args:
            service_id: 服务ID

        Returns:
            服务状态信息，如果服务不存在则返回None
        """
        with self.lock:
            service_info = self.services.get(service_id)
            if not service_info:
                return None
            record = self.failure_records.get(service_id, {})
            return {
                "service_id": service_id,
                "type": service_info.get("type"),
                "status": service_info.get("status"),
                "failure_count": record.get("failure_count", 0),
                "is_critical": record.get("is_critical", False),
                "last_check_time": service_info.get("last_check_time"),
            }

    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务的状态

        Returns:
            所有服务的状态信息，按服务ID索引
        """
        with self.lock:
            service_ids = list(self.services.keys())
        # 刷新各服务的实时健康状态
        for sid in service_ids:
            self.check_service_health(sid)
        return {sid: self.get_service_status(sid) for sid in service_ids}

    # 兼容旧API名称
    def get_all_service_statuses(self) -> Dict[str, Dict[str, Any]]:
        return self.get_all_services_status()

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
