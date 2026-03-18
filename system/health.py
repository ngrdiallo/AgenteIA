"""
Health monitoring: controlla stato risorse, storage, modelli.
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Snapshot stato del sistema."""
    status: str  # healthy | degraded | unhealthy
    uptime_s: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    backends_available: int = 0
    documents_count: int = 0
    checks: Dict[str, bool] = field(default_factory=dict)


class HealthMonitor:
    """Monitor risorse e stato del sistema."""

    def __init__(self):
        self._start_time = time.time()

    def check(self) -> HealthStatus:
        """Esegui tutti i controlli di salute."""
        checks = {}

        # Disk space
        try:
            disk = shutil.disk_usage(str(settings.STORAGE_DIR))
            disk_free = disk.free / (1024 ** 3)
            disk_total = disk.total / (1024 ** 3)
            checks["disk_ok"] = disk_free > 1.0  # Almeno 1 GB libero
        except Exception:
            disk_free, disk_total = 0.0, 0.0
            checks["disk_ok"] = False

        # Directory scrivibili
        for name, path in [
            ("storage", settings.STORAGE_DIR),
            ("logs", settings.LOGS_DIR),
            ("temp", settings.TEMP_DIR),
        ]:
            try:
                test_file = path / ".health_check"
                test_file.write_text("ok")
                test_file.unlink()
                checks[f"{name}_writable"] = True
            except Exception:
                checks[f"{name}_writable"] = False

        # Backend LLM disponibili
        backends_available = sum(1 for k in [
            settings.OPENROUTER_API_KEY,
            settings.GROQ_API_KEY,
            settings.MISTRAL_API_KEY,
            settings.GOOGLE_API_KEY,
            settings.HF_TOKEN,
            settings.DEEPSEEK_API_KEY,
        ] if k)
        checks["backends_ok"] = backends_available >= 1

        # Stato complessivo
        critical_ok = checks.get("disk_ok", False) and checks.get("storage_writable", False)
        if critical_ok and backends_available >= 2:
            status = "healthy"
        elif critical_ok and backends_available >= 1:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthStatus(
            status=status,
            uptime_s=round(time.time() - self._start_time, 1),
            disk_free_gb=round(disk_free, 1),
            disk_total_gb=round(disk_total, 1),
            backends_available=backends_available,
            checks=checks,
        )
