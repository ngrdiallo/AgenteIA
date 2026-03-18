"""
Cleanup manager: pulizia automatica file temporanei e log vecchi.
PORTED FROM: AgenteIA-Production/src/cleanup_manager.py (refactored)
"""

import logging
import os
import shutil
import threading
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class CleanupManager:
    """
    Gestione pulizia automatica:
    - File temporanei (temp_uploads) più vecchi di N ore
    - Log più vecchi di N giorni
    - Eseguito periodicamente in background
    """

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self) -> None:
        """Avvia il cleanup periodico in background."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cleanup")
        self._thread.start()
        logger.info(f"♻️ CleanupManager avviato (intervallo: {settings.CLEANUP_INTERVAL}s)")

    def stop(self) -> None:
        """Ferma il cleanup."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def run_once(self) -> dict:
        """Esegui un ciclo di cleanup e ritorna statistiche."""
        stats = {"temp_deleted": 0, "logs_deleted": 0, "bytes_freed": 0}
        stats["temp_deleted"], freed1 = self._clean_temp()
        stats["logs_deleted"], freed2 = self._clean_logs()
        stats["bytes_freed"] = freed1 + freed2
        logger.info(
            f"♻️ Cleanup: {stats['temp_deleted']} temp, "
            f"{stats['logs_deleted']} log, "
            f"{stats['bytes_freed'] / 1024:.1f} KB liberati"
        )
        return stats

    def _loop(self) -> None:
        """Loop del daemon di cleanup."""
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Errore cleanup: {e}")
            # Sleep in piccoli intervalli per shutdown rapido
            for _ in range(int(settings.CLEANUP_INTERVAL)):
                if not self._running:
                    break
                time.sleep(1)

    def _clean_temp(self) -> tuple:
        """Elimina file temporanei più vecchi della soglia."""
        deleted = 0
        freed = 0
        max_age_s = settings.TEMP_MAX_AGE_HOURS * 3600
        now = time.time()

        temp_dir = settings.TEMP_DIR
        if not temp_dir.exists():
            return 0, 0

        for item in temp_dir.iterdir():
            try:
                age = now - item.stat().st_mtime
                if age > max_age_s:
                    size = item.stat().st_size
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                    deleted += 1
                    freed += size
            except Exception as e:
                logger.debug(f"Impossibile eliminare {item}: {e}")

        return deleted, freed

    def _clean_logs(self) -> tuple:
        """Elimina file di log più vecchi della soglia."""
        deleted = 0
        freed = 0
        max_age_s = settings.LOG_MAX_AGE_DAYS * 86400
        now = time.time()

        logs_dir = settings.LOGS_DIR
        if not logs_dir.exists():
            return 0, 0

        for item in logs_dir.glob("*.log*"):
            try:
                age = now - item.stat().st_mtime
                if age > max_age_s:
                    size = item.stat().st_size
                    item.unlink()
                    deleted += 1
                    freed += size
            except Exception as e:
                logger.debug(f"Impossibile eliminare log {item}: {e}")

        return deleted, freed
