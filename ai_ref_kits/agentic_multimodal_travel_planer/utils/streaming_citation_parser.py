"""Streaming Citation Parser.

This module provides a state machine parser for extracting markdown citations
from a streaming text input.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class Citation:
    """Data class representing a parsed citation."""
    url: str
    title: str
    description: str
    start_index: int
    end_index: int


class State(Enum):
    """Enum representing the states of the citation parser."""
    INITIAL = "initial"
    LINK_TEXT = "link_text"
    LINK_MIDDLE = "link_middle"
    LINK_LOCATION = "link_location"
    DONE = "done"


class StreamingCitationParser:
    """
    State machine parser that extracts markdown citations while streaming.

    Streams clean text immediately and extracts citation metadata when complete
    links are detected. Ignores image markdown (![alt](url)).
    """

    def __init__(self):
        """Initialize the parser state."""
        self.buffer = ""
        self.state = State.INITIAL
        self.maybe_link_start = 0
        self.link_text = ""
        self.link_url = ""
        self.citations = []
        self.clean_position = 0  # Position in the clean (output) text

    def process_chunk(self, chunk: str) -> tuple[str, list[Citation]]:
        """
        Process a chunk of text through the state machine.

        Args:
            chunk: A string chunk of text to process.

        Returns:
            Tuple of (clean_text_to_stream, new_citations)
        """
        self.buffer += chunk
        output = ""
        new_citations = []

        i = self.maybe_link_start

        while i < len(self.buffer):
            char = self.buffer[i]

            if self.state == State.INITIAL:
                if char == "[":
                    # Check if this is an image link ![...]
                    if i > 0 and self.buffer[i - 1] == "!":
                        # This is an image, skip parsing it as a citation
                        i += 1
                        continue

                    # Found potential link start
                    # Stream everything before this point
                    output += self.buffer[self.maybe_link_start:i]
                    self.maybe_link_start = i
                    self.link_text = ""
                    self.link_url = ""
                    self.state = State.LINK_TEXT
                    i += 1
                else:
                    i += 1

            elif self.state == State.LINK_TEXT:
                if char == "]":
                    self.state = State.LINK_MIDDLE
                    i += 1
                elif char == "\n":
                    # Newline breaks the link, back to initial
                    self.state = State.INITIAL
                    self.maybe_link_start = i
                elif char == "[":
                    # Nested bracket, restart
                    output += self.buffer[self.maybe_link_start:i]
                    self.maybe_link_start = i
                    self.link_text = ""
                    i += 1
                else:
                    self.link_text += char
                    i += 1

            elif self.state == State.LINK_MIDDLE:
                if char == "(":
                    self.state = State.LINK_LOCATION
                    i += 1
                else:
                    # Not a link after all, back to initial
                    self.state = State.INITIAL
                    self.maybe_link_start = i

            elif self.state == State.LINK_LOCATION:
                if char == ")":
                    # Complete link found!
                    self.state = State.DONE
                    i += 1
                    break  # Process the complete link
                elif char == "\n":
                    # Newline breaks the link
                    self.state = State.INITIAL
                    self.maybe_link_start = i
                else:
                    self.link_url += char
                    i += 1

        # Handle DONE state - we found a complete link
        if self.state == State.DONE:
            # Calculate citation position BEFORE adding link text to output
            citation_start = self.clean_position + len(output)
            citation_end = citation_start + len(self.link_text)

            # Stream the link text only (not the markdown syntax)
            output += self.link_text

            new_citations.append(
                Citation(
                    url=self.link_url,
                    title=(
                        self.link_url.split("/")[-1].replace("-", " ").title()
                        or self.link_text[:50]
                    ),
                    description=(
                        self.link_text[:100] +
                        ("..." if len(self.link_text) > 100 else "")
                    ),
                    start_index=citation_start,
                    end_index=citation_end,
                )
            )

            self.citations.extend(new_citations)

            # Drop the processed markdown link from buffer
            self.buffer = self.buffer[i:]

            # Reset for next link
            self.state = State.INITIAL
            self.maybe_link_start = 0
            self.link_text = ""
            self.link_url = ""
        else:
            # Keep unprocessed part in buffer
            # Only output up to maybe_link_start if we're in middle of parsing
            if self.state == State.INITIAL:
                # Can safely output everything processed
                if i > self.maybe_link_start:
                    output += self.buffer[self.maybe_link_start:i]
                    self.buffer = self.buffer[i:]
                    self.maybe_link_start = 0
            else:
                # In middle of parsing a potential link
                # Keep buffer from maybe_link_start onwards
                self.buffer = self.buffer[self.maybe_link_start:]
                self.maybe_link_start = 0

        # Update clean position
        self.clean_position += len(output)

        return output, new_citations

    def finalize(self) -> str:
        """
        Process any remaining buffer content.

        Returns:
            Tuple of (remaining_clean_text, all_citations)
        """
        output = ""

        # If we're in middle of parsing, treat it as regular text
        if self.state != State.INITIAL and self.buffer:
            output = self.buffer
            self.buffer = ""
        elif self.state == State.INITIAL and self.buffer:
            output = self.buffer[self.maybe_link_start:]
            self.buffer = ""

        self.state = State.INITIAL
        self.maybe_link_start = 0

        return output

    def reset(self):
        """Reset parser state for reuse."""
        self.buffer = ""
        self.state = State.INITIAL
        self.maybe_link_start = 0
        self.link_text = ""
        self.link_url = ""
        self.citations = []
        self.clean_position = 0
