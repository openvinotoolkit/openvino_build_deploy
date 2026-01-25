"""
Agent Scratchpad Tool - Allows agents to track their reasoning and actions.

This tool provides a working memory (scratchpad) where agents can:
- Record actions they've taken
- Store observations/results from tools
- Review their previous reasoning
- Avoid repeating actions
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field
from beeai_framework.tools import Tool, ToolRunOptions, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter

logger = logging.getLogger(__name__)

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
    content: Optional[str] = Field(
        default=None,
        description=(
            "Content to write/append (required for 'write' and 'append' "
            "operations)"
        ),
    )


class ScratchpadTool(Tool):
    """Tool for managing agent scratchpad (working memory)."""

    _scratchpads: Dict[str, list] = {}

    def __init__(self, session_id: Optional[str] = None):
        """Initialize scratchpad tool.

        Args:
            session_id: Optional session identifier (deprecated, not used).
                        Session ID is now extracted from RunContext.
        """
        super().__init__()
        self.middlewares = []

    @staticmethod
    def _ensure_session(session_id: str) -> None:
        """Ensure a session exists in scratchpads."""
        if session_id not in ScratchpadTool._scratchpads:
            ScratchpadTool._scratchpads[session_id] = []

    def _get_session_id(self, context: Optional[RunContext] = None) -> str:
        """Extract session ID from context.

        Always returns "default" to maintain a single persistent scratchpad
        across all requests, ensuring information is retained between interactions.

        Args:
            context: Run context (not used, maintained for compatibility).

        Returns:
            Session ID string (always "default").
        """
        # Use a single persistent session for all operations
        # This ensures the scratchpad persists across HTTP requests
        return "default"

    def _log_operation(self, operation: str, session_id: str,
                       content: Optional[str] = None,
                       result: Optional[str] = None) -> None:
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
    def input_schema(self) -> type[BaseModel]:
        """Input schema for the tool."""
        return ScratchpadInput

    def _create_emitter(self) -> Emitter:
        """Create emitter for the tool."""
        return Emitter()

    def _get_entries(self, session_id: str) -> list:
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
            Formatted scratchpad content string with validation warnings.
        """
        entries = self._get_entries(session_id)
        if not entries:
            result = "Scratchpad is empty. No actions recorded yet."
            logger.info(f"ScratchpadTool[{session_id}]: READ - Empty")
            self._log_operation("read", session_id, result=result)
            return result

        result = "=== AGENT SCRATCHPAD ===\n\n"
        result += "\n\n".join(f"[{i}] {entry}" for i, entry in
                              enumerate(entries, 1))
        
        # Add validation warnings for missing required fields
        entry_text = entries[0] if entries else ""
        warnings = []
        
        # Check if this looks like flight data (has departure_date or return_date but missing departure_city)
        if ('departure_date:' in entry_text or 'return_date:' in entry_text) and 'departure_city:' not in entry_text:
            warnings.append("\n\n⚠️ WARNING: departure_city is MISSING for flights! You MUST ask: 'What is your departure city?' before proceeding.")
        
        # Check if this looks like hotel data (has check_in_date but missing guests)
        if 'check_in_date:' in entry_text and 'guests:' not in entry_text:
            warnings.append("\n\n⚠️ WARNING: guests field is MISSING for hotels! Add 'guests: 1' as default.")
        
        result += "".join(warnings)
        
        logger.info(f"ScratchpadTool[{session_id}]: READ - {len(entries)} entries" + 
                   (f" with {len(warnings)} warnings" if warnings else ""))
        self._log_operation("read", session_id,
                           result=f"{len(entries)} entries")
        return result

    @staticmethod
    def _parse_key_value_pairs(content: str) -> dict:
        """Parse key-value pairs from scratchpad content.

        Handles formats like:
        - "key: value"
        - "key1: value1, key2: value2"
        - "key: value, key2: value2, key3: value3"

        Args:
            content: Content string to parse.

        Returns:
            Dictionary of key-value pairs.
        """
        pairs = {}
        # Split by comma, but be careful with commas inside values
        parts = [p.strip() for p in content.split(',')]
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    pairs[key] = value
        return pairs

    @staticmethod
    def _merge_entries(entries: list, new_pairs: dict) -> list:
        """Merge new key-value pairs into existing entries.

        Args:
            entries: List of existing scratchpad entries.
            new_pairs: Dictionary of new key-value pairs to merge.

        Returns:
            Updated list of entries (consolidated).
        """
        # Parse all existing entries into a single dict
        consolidated = {}
        for entry in entries:
            pairs = ScratchpadTool._parse_key_value_pairs(entry)
            consolidated.update(pairs)

        # Merge new pairs (new values override old ones)
        consolidated.update(new_pairs)

        # Convert back to entry format
        if consolidated:
            # Create a single consolidated entry
            entry_str = ', '.join(f"{k}: {v}" for k, v in consolidated.items())
            return [entry_str]
        return []

    def _write_scratchpad(self, entry: str, session_id: str) -> str:
        """Add or update entry in the scratchpad.

        Merges key-value pairs with existing entries to avoid duplicates.
        If entry contains key-value pairs (format: "key: value"), it will
        update existing entries with the same keys.
        
        Automatically adds missing date field aliases to support both flights
        and hotels:
        - If check_in_date exists, adds departure_date with same value
        - If departure_date exists, adds check_in_date with same value
        - If check_out_date exists, adds return_date with same value
        - If return_date exists, adds check_out_date with same value

        Args:
            entry: Content to add/update.
            session_id: Session identifier.

        Returns:
            Success message.
        """
        entries = self._get_entries(session_id)
        new_pairs = self._parse_key_value_pairs(entry)

        if new_pairs:
            # Automatically add missing date field aliases
            if 'check_in_date' in new_pairs and 'departure_date' not in new_pairs:
                new_pairs['departure_date'] = new_pairs['check_in_date']
                logger.info(f"ScratchpadTool: Auto-added departure_date = {new_pairs['departure_date']}")
            
            if 'departure_date' in new_pairs and 'check_in_date' not in new_pairs:
                new_pairs['check_in_date'] = new_pairs['departure_date']
                logger.info(f"ScratchpadTool: Auto-added check_in_date = {new_pairs['check_in_date']}")
            
            if 'check_out_date' in new_pairs and 'return_date' not in new_pairs:
                new_pairs['return_date'] = new_pairs['check_out_date']
                logger.info(f"ScratchpadTool: Auto-added return_date = {new_pairs['return_date']}")
            
            if 'return_date' in new_pairs and 'check_out_date' not in new_pairs:
                new_pairs['check_out_date'] = new_pairs['return_date']
                logger.info(f"ScratchpadTool: Auto-added check_out_date = {new_pairs['check_out_date']}")
            
            # Merge with existing entries
            entries[:] = self._merge_entries(entries, new_pairs)
            result = f"Updated scratchpad: {', '.join(f'{k}: {v}' for k, v in new_pairs.items())}"
        else:
            # If no key-value pairs found, append as-is (for non-structured entries)
            entries.append(entry)
            result = f"Added to scratchpad: {entry}"

        logger.info(f"ScratchpadTool[{session_id}]: WRITE - "
                   f"{len(entries)} total entries")
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
            self._log_operation("append", session_id, content=text,
                               result=result)
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
        logger.info(f"ScratchpadTool[{session_id}]: CLEAR - "
                    f"{entries_count} entries")
        self._log_operation("clear", session_id,
                           result=f"Cleared {entries_count} entries")
        return result

    async def _run(
        self,
        input: ScratchpadInput,
        options: Optional[ToolRunOptions] = None,
        context: Optional[RunContext] = None,
    ) -> StringToolOutput:
        """Execute scratchpad operation.

        Args:
            input: ScratchpadInput model instance.
            options: Optional tool run options.
            context: Optional run context.

        Returns:
            StringToolOutput with the result of the operation.
        """
        # Get session ID (always "default" for persistent storage)
        session_id = self._get_session_id(context)
        operation = input.operation.lower().strip()
        content = input.content

        logger.info(f"ScratchpadTool[{session_id}]: operation='{operation}', "
                   f"content='{content}'")

        if not operation:
            error_msg = (
                "Error: 'operation' parameter is required. "
                "Use 'read', 'write', 'append', or 'clear'."
            )
            return StringToolOutput(result=error_msg)

        # Operation handlers
        handlers = {
            "read": lambda: self._read_scratchpad(session_id),
            "write": lambda: (
                self._write_scratchpad(content, session_id)
                if content
                else "Error: 'write' operation requires 'content' parameter."
            ),
            "append": lambda: (
                self._append_scratchpad(content, session_id)
                if content
                else "Error: 'append' operation requires 'content' parameter."
            ),
            "clear": lambda: self._clear_scratchpad(session_id),
        }

        handler = handlers.get(operation)
        if handler:
            result = handler()
            return StringToolOutput(result=result)

        error_msg = (
            f"Unknown operation: {operation}. "
            "Use 'read', 'write', 'append', or 'clear'."
        )
        return StringToolOutput(result=error_msg)

    @classmethod
    def get_scratchpad_for_session(cls, session_id: str) -> list:
        """Get scratchpad entries for a specific session.

        Args:
            session_id: Session identifier.

        Returns:
            List of scratchpad entries.
        """
        return cls._scratchpads.get(session_id, [])

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        """Clear scratchpad for a specific session.

        Args:
            session_id: Session identifier.
        """
        if session_id in cls._scratchpads:
            cls._scratchpads[session_id] = []
