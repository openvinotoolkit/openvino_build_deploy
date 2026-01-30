# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""
Agent Scratchpad Tool - Allows agents to track their reasoning and actions.

This tool provides a working memory (scratchpad) where agents can:
- Record actions they've taken
- Store observations/results from tools
- Review their previous reasoning
- Avoid repeating actions
"""

import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools import (
    StringToolOutput,
    Tool,
    ToolInputValidationError,
    ToolRunOptions,
)
from pydantic import BaseModel, Field

logger = Logger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"
SCRATCHPAD_LOG_FILE = LOGS_DIR / "scratchpad.log"
LOGS_DIR.mkdir(exist_ok=True)


class ScratchpadInput(BaseModel):
    """Input schema for scratchpad operations."""

    operation: str = Field(
        description=(
            "Operation to perform: 'read' to view scratchpad, "
            "'write' to add entry, 'append' to add to last entry, "
            "'clear' to reset"
        ),
        enum=["read", "write", "append", "clear"],
    )
    content: str | None = Field(
        default=None,
        description=(
            "Content to write/append (required for 'write' and 'append' " "operations)"
        ),
    )


class ScratchpadTool(Tool):
    """Tool for managing agent scratchpad (working memory).

    Supports two types of content:
    1. Key-Value Pairs: Automatically consolidated into a single entry
       - Format: "key: value, another_key: another_value"
       - Duplicate keys are updated with latest values
       - Results in ONE entry containing all key-value pairs
       - Special handling for travel dates: automatically creates aliases
         (departure_date ↔ check_in_date, return_date ↔ check_out_date)

    2. Plain Text: Appended as separate entries
       - Format: Any text without colons
       - Each write creates a new list entry

    This design allows structured state management (key-value) while
    preserving free-form notes (plain text) as separate items.

    Design Note (Lifecycle Management):
    This tool uses a class-level dictionary (`_scratchpads`) for storage. In a
    long-running process, this dictionary can grow indefinitely. Consumers
    should ensure that `clear_session(session_id)` is called when a session
    or agent run is complete to prevent memory leaks. For distributed
    deployments, consider a persistent external store instead.
    """

    _scratchpads: ClassVar[dict[str, list[str]]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _allowed_keys: ClassVar[set[str]] = {
        "destination",
        "departure_city",
        "departure_date",
        "return_date",
        "check_in_date",
        "check_out_date",
        "guests",
        "class",
        "intent",
        "mode",
        "status",
        "confirmation",
        "confirmed",
        "previous_handoff",
        "previous_result",
        "previous_handoff_result",
        "handoff_result",
    }

    def __init__(self) -> None:
        """Initialize scratchpad tool."""
        super().__init__()
        self.middlewares = []

    @classmethod
    def _ensure_session(cls, session_id: str) -> None:
        """Ensure a session exists in scratchpads."""
        if session_id not in cls._scratchpads:
            cls._scratchpads[session_id] = []

    def _get_session_id(self, context: RunContext | None = None) -> str:
        """Extract session ID from context for each tool invocation.

        Each conversation gets its own scratchpad, automatically isolated
        from other conversations. This eliminates the need for manual clearing
        between travel planning sessions.

        The session ID is extracted fresh from the RunContext on each call,
        ensuring that different conversations (with different contexts) get
        different scratchpads.

        Args:
            context: Run context to extract session identifier from.

        Returns:
            Session ID string for data isolation.
        """
        session_id = None

        # Prefer A2A context_id (stable across messages in the same conversation)
        try:
            from beeai_framework.adapters.a2a.serve import context as a2a_context

            a2a_ctx = a2a_context._storage.get(None)
            if a2a_ctx and hasattr(a2a_ctx, 'context_id') and a2a_ctx.context_id:
                session_id = str(a2a_ctx.context_id)
        except Exception:
            pass

        # Fallback to group_id if A2A context not available
        if not session_id and context and hasattr(context, "group_id") and context.group_id:
            session_id = str(context.group_id)

        # Generate unique session ID if none found
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:16]}"
            logger.warning(f"No stable session ID available, generated: {session_id}")

        logger.debug(f"Scratchpad using session: {session_id}")
        return session_id

    def _log_operation(
        self,
        operation: str,
        session_id: str,
        content: str | None = None,
        result: str | None = None,
    ) -> None:
        """Write scratchpad operation to log file.

        Logs only the current state of the scratchpad for this session,
        overwriting previous entries for the same session.

        Args:
            operation: Operation name (read, write, append, clear).
            session_id: Session identifier.
            content: Optional content that was written/appended.
            result: Optional result message.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entries = self._scratchpads.get(session_id, [])

            # Read existing log file
            log_lines = []
            if SCRATCHPAD_LOG_FILE.exists():
                with open(SCRATCHPAD_LOG_FILE, "r", encoding="utf-8") as f:
                    log_lines = f.readlines()

            # Remove old entries for this session (keep other sessions)
            filtered_lines = []
            skip_until_separator = False
            for i, line in enumerate(log_lines):
                # Check if this line starts a new session entry
                if line.startswith("[") and "Session:" in line:
                    # If this is our session, mark to skip until separator
                    if session_id in line:
                        skip_until_separator = True
                    else:
                        # Different session, keep it
                        skip_until_separator = False
                        filtered_lines.append(line)
                elif skip_until_separator:
                    # Skip lines until we hit the separator
                    if line.strip() == "-" * 80:
                        skip_until_separator = False
                else:
                    # Keep this line
                    filtered_lines.append(line)

            # Append new entry for this session
            with open(SCRATCHPAD_LOG_FILE, "w", encoding="utf-8") as f:
                f.writelines(filtered_lines)
                f.write(f"\n[{timestamp}] Session: {session_id}\n")
                f.write(f"Operation: {operation}\n")
                if content:
                    f.write(f"Content: {content}\n")
                if result:
                    f.write(f"Result: {result}\n")
                f.write(f"Current scratchpad ({len(entries)} entries):\n")
                for i, entry in enumerate(entries, 1):
                    f.write(f"  [{i}] {entry}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to write to scratchpad log file: {e}")

    @property
    def name(self) -> str:
        """Tool name."""
        return "scratchpad"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "Manage your working memory (scratchpad). Use this to track "
            "what you've done, what results you got, and avoid repeating "
            "actions. Operations: 'read' - see your scratchpad, 'write' - "
            "add an entry, 'clear' - reset scratchpad, 'append' - add to "
            "existing entry."
        )

    @property
    def input_schema(self) -> type[ScratchpadInput]:
        """Input schema for the tool."""
        return ScratchpadInput

    def _create_emitter(self) -> Emitter:
        """Create emitter for the tool."""
        return Emitter.root().child(
            namespace=["tool", "scratchpad"],
            creator=self,
        )

    def _get_entries(self, session_id: str) -> list[str]:
        """Get scratchpad entries for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of scratchpad entries.
        """
        self._ensure_session(session_id)
        return self._scratchpads[session_id]

    def _read_scratchpad(self, session_id: str) -> str:
        """Read the current scratchpad content.

        Args:
            session_id: Session identifier.

        Returns:
            Formatted scratchpad content string.
        """
        entries = self._get_entries(session_id)
        if not entries:
            result = "Scratchpad is empty. No actions recorded yet."
            logger.info(f"ScratchpadTool[{session_id}]: READ - Empty")
            self._log_operation("read", session_id, result=result)
            return result

        result = "=== AGENT SCRATCHPAD ===\n\n"
        result += "\n\n".join(f"[{i}] {entry}" for i, entry in enumerate(entries, 1))

        logger.info(f"ScratchpadTool[{session_id}]: READ - {len(entries)} entries")
        self._log_operation("read", session_id, result=f"{len(entries)} entries")
        return result

    @staticmethod
    def _parse_key_value_pairs(content: str) -> dict:
        """Parse key-value pairs from scratchpad content.

        Uses regex to correctly handle values containing commas, which prevents
        incorrectly splitting "item: milk, bread, eggs" into separate entries.

        Format Examples:
        - Simple: "key: value" → {"key": "value"}
        - Multiple: "key1: val1, key2: val2" → {"key1": "val1", "key2": "val2"}
        - Comma in value: "item: milk, bread, key2: val2" → {"item": "milk, bread", "key2": "val2"}
        - Hyphenated keys: "Content-Type: json, user-id: 123" → {"Content-Type": "json", "user-id": "123"}

        Implementation Note:
        A simple split-by-comma approach fails when values contain commas.
        The regex pattern works by:
        1. Matching any characters except ':' as the key: ([^:]+)
        2. Matching the colon separator: :
        3. Capturing everything until the next "key:" pattern or end: (.*?)(?=...)

        This ensures commas within values are preserved while correctly
        identifying multiple key-value pairs separated by commas.

        Args:
            content: Content string to parse.

        Returns:
            Dictionary of key-value pairs.
        """
        pairs = {}
        special_result_keys = ("handoff_result", "previous_handoff_result", "previous_result")

        # If a result key is present, treat it as the final field and capture
        # the rest of the content verbatim to avoid splitting on commas/colons.
        lower_content = content.lower()
        special_match = None
        for key in special_result_keys:
            pattern_key = re.compile(rf"(^|,\\s*){re.escape(key)}\\s*:", re.IGNORECASE)
            match = pattern_key.search(content)
            if match:
                key_start = match.start() + len(match.group(1))
                value_start = match.end()
                special_match = (key, key_start, value_start)
                break

        if special_match:
            result_key, start_idx, value_start = special_match
            prefix = content[:start_idx].strip().rstrip(",")
            result_value = content[value_start:].strip()
            if prefix:
                pairs.update(ScratchpadTool._parse_key_value_pairs(prefix))
            if result_value:
                pairs[result_key] = result_value
            return pairs

        # Regex breakdown:
        # ([^:]+)           - Capture key (any chars except colon)
        # :\s*              - Match colon and optional whitespace
        # (.*?)             - Capture value (non-greedy)
        # (?=               - Lookahead (doesn't consume characters):
        #   \s*,\s*[^:]+:   - Next key-value pair (comma, then key:)
        #   |               - OR
        #   \s*$            - End of string
        # )
        pattern = re.compile(r"([^:]+):\s*(.*?)(?=\s*,\s*[^:]+:|\s*$)")

        for match in pattern.finditer(content):
            key = match.group(1).strip().strip(" ,")
            value = match.group(2).strip()
            # Remove trailing commas and extra whitespace from values
            value = value.rstrip(", ").strip()
            if not key or not value:
                continue

            key_lower = key.lower()
            if key_lower not in ScratchpadTool._allowed_keys:
                continue

            pairs[key_lower] = value

        return pairs

    @staticmethod
    def _merge_entries(entries: list[str], new_pairs: dict) -> list[str]:
        """Merge new key-value pairs into existing entries.

        IMPORTANT BEHAVIOR:
        This method consolidates ALL key-value pairs (existing + new) into a
        SINGLE entry. This means the returned list will contain at most ONE
        consolidated entry, not multiple separate entries.

        Design Rationale:
        - Key-value pairs represent structured state that should be merged
        - Each key appears only once with its latest value
        - Prevents duplicate keys and maintains a single source of truth
        - Example: Writing "city: Boston" then "date: 2025-01-28" results in
          ONE entry: "city: Boston, date: 2025-01-28"

        This is different from non-key-value entries (plain text) which are
        appended as separate list items.

        Args:
            entries: List of existing scratchpad entries.
            new_pairs: Dictionary of new key-value pairs to merge.

        Returns:
            Updated list with a SINGLE consolidated entry containing all
            key-value pairs (old + new), with new values overriding old
            values for duplicate keys. Returns empty list if no valid pairs.
        """
        # Parse all existing entries into a single dict
        consolidated = {}
        for entry in entries:
            pairs = ScratchpadTool._parse_key_value_pairs(entry)
            consolidated.update(pairs)

        # Merge new pairs (new values override old ones for duplicate keys)
        consolidated.update(new_pairs)

        # Convert back to entry format
        if consolidated:
            # Create a single consolidated entry containing all pairs
            entry_str = ", ".join(f"{k}: {v}" for k, v in consolidated.items())
            return [entry_str]
        return []

    def _write_scratchpad(self, entry: str, session_id: str) -> str:
        """Add or update entry in the scratchpad.

        Behavior depends on entry format:

        1. Key-Value Pairs (contains ":"):
           - Parsed and merged with existing key-value pairs
           - Results in a SINGLE consolidated entry
           - New values override old values for duplicate keys
           - Special travel planner feature: Automatically adds date field aliases
             to support both flights and hotels:
             * check_in_date ↔ departure_date
             * check_out_date ↔ return_date

        2. Plain Text (no ":"):
           - Appended as a new separate entry
           - Multiple plain text entries can exist

        This design ensures structured state (key-value) remains consolidated
        while allowing free-form notes (plain text) to accumulate.

        Args:
            entry: Content to add/update.
            session_id: Session identifier.

        Returns:
            Success message describing the action taken.
        """
        entries = self._get_entries(session_id)
        new_pairs = self._parse_key_value_pairs(entry)

        if new_pairs:
            # Travel planner specific: Automatically add missing date field aliases
            if "check_in_date" in new_pairs and "departure_date" not in new_pairs:
                new_pairs["departure_date"] = new_pairs["check_in_date"]
                logger.info(
                    f"ScratchpadTool: Auto-added departure_date = "
                    f"{new_pairs['departure_date']}"
                )

            if "departure_date" in new_pairs and "check_in_date" not in new_pairs:
                new_pairs["check_in_date"] = new_pairs["departure_date"]
                logger.info(
                    f"ScratchpadTool: Auto-added check_in_date = "
                    f"{new_pairs['check_in_date']}"
                )

            if "check_out_date" in new_pairs and "return_date" not in new_pairs:
                new_pairs["return_date"] = new_pairs["check_out_date"]
                logger.info(
                    f"ScratchpadTool: Auto-added return_date = "
                    f"{new_pairs['return_date']}"
                )

            if "return_date" in new_pairs and "check_out_date" not in new_pairs:
                new_pairs["check_out_date"] = new_pairs["return_date"]
                logger.info(
                    f"ScratchpadTool: Auto-added check_out_date = "
                    f"{new_pairs['check_out_date']}"
                )

            # Merge with existing entries
            entries[:] = self._merge_entries(entries, new_pairs)
            # Since we just merged new_pairs (which is not empty), entries will have content
            result = f"Updated scratchpad to: {entries[0]}"
        else:
            # If no key-value pairs found, append as-is (for non-structured entries)
            entries.append(entry)
            result = f"Added to scratchpad: {entry}"

        logger.info(
            f"ScratchpadTool[{session_id}]: WRITE - " f"{len(entries)} total entries"
        )
        self._log_operation("write", session_id, content=entry, result=result)
        return result

    def _append_scratchpad(self, text: str, session_id: str) -> str:
        """Append to the last entry in scratchpad.

        Args:
            text: Text to append.
            session_id: Session identifier.

        Returns:
            Success or error message.
        """
        entries = self._get_entries(session_id)
        if not entries:
            result = "No entry to append to. Use 'write' first."
            logger.info(f"ScratchpadTool[{session_id}]: APPEND - No entries")
            self._log_operation("append", session_id, content=text, result=result)
            return result

        entries[-1] += f" {text}"
        result = f"Appended to last entry: {text}"
        logger.info(f"ScratchpadTool[{session_id}]: APPEND - Updated")
        self._log_operation("append", session_id, content=text, result=result)
        return result

    def _clear_scratchpad(self, session_id: str) -> str:
        """Clear the scratchpad.

        Args:
            session_id: Session identifier.

        Returns:
            Success message.
        """
        entries_count = len(self._get_entries(session_id))
        self._scratchpads[session_id] = []
        result = "Scratchpad cleared."
        logger.info(
            f"ScratchpadTool[{session_id}]: CLEAR - " f"{entries_count} entries"
        )
        self._log_operation("clear", session_id, result=f"Cleared {entries_count} entries")
        return result

    async def _run(
        self,
        input: ScratchpadInput,
        options: ToolRunOptions | None = None,
        context: RunContext | None = None,
    ) -> StringToolOutput:
        """Execute scratchpad operation.

        Args:
            input: ScratchpadInput model instance.
            options: Optional tool run options.
            context: Optional run context.

        Returns:
            StringToolOutput with the result of the operation.
        """
        session_id = self._get_session_id(context)
        operation = input.operation.lower().strip()
        content = input.content

        logger.info(
            f"ScratchpadTool[{session_id}]: operation='{operation}', "
            f"content='{content}'"
        )

        result = None
        async with ScratchpadTool._lock:
            if operation in ("write", "append") and not content:
                self._raise_input_validation_error(
                    f"'{operation}' operation requires 'content' parameter."
                )

            handlers = {
                "read": lambda: self._read_scratchpad(session_id),
                "write": lambda: self._write_scratchpad(content, session_id),
                "append": lambda: self._append_scratchpad(content, session_id),
                "clear": lambda: self._clear_scratchpad(session_id),
            }

            # Operation is validated by Pydantic enum, so key existence is guaranteed
            result = handlers[operation]()

        return StringToolOutput(result=result)

    def _raise_input_validation_error(self, message: str) -> None:
        """Raise a ToolInputValidationError with the given message.

        Args:
            message: Error message to include in the exception.

        Raises:
            ToolInputValidationError: Always raised with the provided message.
        """
        raise ToolInputValidationError(message)

    @classmethod
    def get_scratchpad_for_session(cls, session_id: str) -> list[str]:
        """Get scratchpad entries for a specific session.

        Args:
            session_id: Session identifier.

        Returns:
            List of scratchpad entries.
        """
        return cls._scratchpads.get(session_id, []).copy()

    @classmethod
    async def clear_session(cls, session_id: str) -> None:
        """Clear scratchpad for a specific session.

        Args:
            session_id: Session identifier.
        """
        async with cls._lock:
            if session_id in cls._scratchpads:
                cls._scratchpads[session_id] = []
