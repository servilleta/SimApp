import redis, json, os, logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ResultStore:
    """Redis-backed store for completed simulation results (or current status)."""
    def __init__(self):
        try:
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            self._redis.ping()
            logger.info("✅ Redis result store connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable for results store, falling back to in-memory: {e}")
            self._redis = None
            self._mem: Dict[str, Dict[str, Any]] = {}

    def _key(self, sim_id: str) -> str:
        return f"simulation:results:{sim_id}"

    def set(self, sim_id: str, data: Dict[str, Any], ttl: int = 86400):
        try:
            if self._redis:
                self._redis.setex(self._key(sim_id), ttl, json.dumps(data))
            else:
                self._mem[sim_id] = data
        except Exception as e:
            logger.error(f"Failed to set results for {sim_id}: {e}")

    def get(self, sim_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self._redis:
                raw = self._redis.get(self._key(sim_id))
                return json.loads(raw) if raw else None
            return self._mem.get(sim_id)
        except Exception as e:
            logger.error(f"Failed to get results for {sim_id}: {e}")
            return None

    def delete(self, sim_id: str) -> bool:
        """Delete simulation result data"""
        try:
            if self._redis:
                result = self._redis.delete(self._key(sim_id))
                return result > 0
            else:
                return self._mem.pop(sim_id, None) is not None
        except Exception as e:
            logger.error(f"Failed to delete results for {sim_id}: {e}")
            return False

_result_store = ResultStore()

def set_result(sim_id: str, data: Dict[str, Any]):
    _result_store.set(sim_id, data)

def get_result(sim_id: str):
    return _result_store.get(sim_id)

def delete_result(sim_id: str):
    return _result_store.delete(sim_id) 