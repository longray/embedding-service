"""ChangeFeed client for SurrealDB v3.2

Provides real-time change notifications for atom, entity, reference tables.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from surrealdb import Surreal  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class ChangeFeedClient:
    """SurrealDB ChangeFeed client

    Usage:
        client = ChangeFeedClient("ws://localhost:18002", "memory_ns", "memory_db")
        await client.connect()
        await client.subscribe_to_changes("atom", on_change)
        await client.start_listening()
    """

    ALLOWED_TABLES = frozenset({"atom", "entity", "reference", "memory"})

    def __init__(
        self,
        url: str = "ws://localhost:18002",
        namespace: str = "memory_ns",
        database: str = "memory_db",
        username: str = "root",  # nosec B107
        password: str = "root",  # nosec B107 # noqa: S107
    ):
        self._url = url
        self._namespace = namespace
        self._database = database
        self._username = username
        self._password = password
        self._db: Surreal | None = None  # type: ignore[reportGeneralTypeIssues]
        self._subscriptions: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = {}
        self._listening = False

    async def connect(self) -> None:
        """Connect to SurrealDB"""
        self._db = Surreal(self._url)
        db = self._db  # Local binding for type narrowing
        await db.connect()  # pyright: ignore[reportOptionalMemberAccess]
        await db.signin({"user": self._username, "pass": self._password})  # pyright: ignore[reportOptionalMemberAccess]
        await db.use(self._namespace, self._database)  # pyright: ignore[reportOptionalMemberAccess]
        logger.info("[ChangeFeed] Connected to %s/%s", self._namespace, self._database)

    async def close(self) -> None:
        """Close connection"""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("[ChangeFeed] Disconnected")

    async def subscribe_to_changes(
        self,
        table: str,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> str:
        """Subscribe to changes on a table

        Args:
            table: Table name (atom, entity, reference)
            callback: Async callback function for change events

        Returns:
            Query UUID
        """
        if not self._db:
            raise RuntimeError("Not connected to SurrealDB")

        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table}")

        # Start LIVE SELECT query
        query_id = await self._db.query(f"LIVE SELECT * FROM {table}")  # nosec B608
        self._subscriptions[query_id] = callback  # pyright: ignore[reportArgumentType]

        logger.info("[ChangeFeed] Subscribed to %s changes (query_id: %s)", table, query_id)
        return query_id  # pyright: ignore[reportReturnType]

    async def unsubscribe(self, query_id: str) -> None:
        """Unsubscribe from changes"""
        if not self._db:
            return

        await self._db.query(f"KILL {query_id}")  # nosec B608 - query_id 由 subscribe_to_changes 返回的 UUID
        if query_id in self._subscriptions:
            del self._subscriptions[query_id]

        logger.info("[ChangeFeed] Unsubscribed from query %s", query_id)

    async def start_listening(self) -> None:
        """Start listening for change events"""
        if not self._db:
            raise RuntimeError("Not connected to SurrealDB")

        self._listening = True
        logger.info("[ChangeFeed] Started listening for changes")

        while self._listening:
            try:
                # In surrealdb-python, live queries are handled differently
                # This is a simplified version - actual implementation depends on SDK version
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[ChangeFeed] Error listening: %s", e)
                await asyncio.sleep(5)

    def stop_listening(self) -> None:
        """Stop listening for changes"""
        self._listening = False
        logger.info("[ChangeFeed] Stopped listening")

    async def get_changefeed_info(self, table: str) -> dict[str, Any]:
        """Get ChangeFeed info for a table"""
        if not self._db:
            raise RuntimeError("Not connected to SurrealDB")

        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table}")

        result = await self._db.query(f"INFO FOR TABLE {table}")  # nosec B608
        return result  # pyright: ignore[reportReturnType]

    async def verify_changefeed_enabled(self, table: str) -> bool:
        """Verify ChangeFeed is enabled for a table

        Args:
            table: Table name

        Returns:
            True if ChangeFeed is enabled
        """
        try:
            info = await self.get_changefeed_info(table)
            # Check if changefeed is in the table info
            return "changefeed" in str(info).lower()
        except Exception as e:
            logger.error("[ChangeFeed] Error verifying: %s", e)
            return False
