"""In-memory task progress tracking for background tasks."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

# In-memory storage for task status
_task_storage: Dict[str, Dict[str, Any]] = {}
_storage_lock = asyncio.Lock()


async def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve task status by task ID.

    Args:
        task_id: Unique task identifier

    Returns:
        Task information dictionary or None if not found
    """
    async with _storage_lock:
        return _task_storage.get(task_id)


async def list_tasks() -> List[Dict[str, Any]]:
    """
    List all tasks in storage.

    Returns:
        List of all task information dictionaries
    """
    async with _storage_lock:
        return list(_task_storage.values())


async def _update_task_storage(task_id: str, **kwargs) -> None:
    """
    Internal helper to update task storage.

    Used by background tasks to write updates.

    Args:
        task_id: Unique task identifier
        **kwargs: Fields to update in task record
    """
    async with _storage_lock:
        if task_id not in _task_storage:
            # Initialize new task
            _task_storage[task_id] = {
                "task_id": task_id,
                "status": "processing",
                "file_id": None,
                "message": "",
                "chunks_processed": 0,
                "total_chunks": 0,
                "created_at": datetime.utcnow(),
                "completed_at": None,
            }

        # Update fields
        _task_storage[task_id].update(kwargs)
