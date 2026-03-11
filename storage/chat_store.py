"""
Persistent chat session storage — file JSON-based.
Salva le conversazioni su disco in modo che persistano tra sessioni e riavvii.
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

CHAT_DIR = settings.STORAGE_DIR / "chats"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


class ChatSession:
    """Singola sessione di chat."""

    def __init__(
        self,
        session_id: str = "",
        title: str = "Nuova conversazione",
        modalita: str = "generale",
        messages: Optional[List[Dict]] = None,
        created_at: str = "",
        updated_at: str = "",
    ):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.title = title
        self.modalita = modalita
        self.messages: List[Dict] = messages or []
        now = datetime.utcnow().isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "modalita": self.modalita,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        return cls(
            session_id=data.get("session_id", ""),
            title=data.get("title", "Nuova conversazione"),
            modalita=data.get("modalita", "generale"),
            messages=data.get("messages", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Aggiunge un messaggio alla sessione."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if metadata:
            msg["metadata"] = metadata
        self.messages.append(msg)
        self.updated_at = datetime.utcnow().isoformat()

    def auto_title(self):
        """Genera titolo automatico dalla prima domanda utente."""
        for msg in self.messages:
            if msg.get("role") == "user":
                text = msg["content"][:60].strip()
                if len(msg["content"]) > 60:
                    text += "…"
                self.title = text
                return
        self.title = "Nuova conversazione"


class ChatStore:
    """Gestisce il salvataggio/caricamento delle sessioni di chat su disco."""

    def __init__(self, chat_dir: Optional[Path] = None):
        self._dir = chat_dir or CHAT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info(f"💬 ChatStore inizializzato: {self._dir}")

    def _session_path(self, session_id: str) -> Path:
        """Path sicuro per un file sessione."""
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe_id}.json"

    def save(self, session: ChatSession) -> None:
        """Salva una sessione su disco."""
        path = self._session_path(session.session_id)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> Optional[ChatSession]:
        """Carica una sessione dal disco."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ChatSession.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Errore lettura sessione {session_id}: {e}")
            return None

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """Lista sessioni ordinate per data aggiornamento (più recenti prima)."""
        sessions = []
        for path in self._dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", path.stem),
                    "title": data.get("title", "Senza titolo"),
                    "modalita": data.get("modalita", "generale"),
                    "message_count": len(data.get("messages", [])),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Ordina per data aggiornamento descrescente
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions[:limit]

    def delete(self, session_id: str) -> bool:
        """Elimina una sessione."""
        path = self._session_path(session_id)
        if path.exists():
            with self._lock:
                path.unlink()
            logger.info(f"🗑️ Sessione chat eliminata: {session_id}")
            return True
        return False

    def delete_all(self) -> int:
        """Elimina tutte le sessioni. Ritorna il numero eliminato."""
        count = 0
        with self._lock:
            for path in self._dir.glob("*.json"):
                path.unlink()
                count += 1
        logger.info(f"🗑️ Eliminate {count} sessioni chat")
        return count
