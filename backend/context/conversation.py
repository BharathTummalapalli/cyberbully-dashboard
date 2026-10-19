"""
Conversational Context Extractor.
Extracts preceding messages, parent replies, conversational turns, and time deltas.
Strictly ensures temporal leakage prevention: only records prior to current_timestamp are queried.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

class ConversationContextExtractor:
    def __init__(self):
        pass

    def extract(
        self,
        text: str,
        author: str,
        target_user: str,
        timestamp: datetime,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        history_records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts conversational context for a given message.

        Args:
            text: Current message text.
            author: Current author identifier.
            target_user: Target recipient identifier.
            timestamp: Timestamp of the current message (reference point for leakage prevention).
            thread_id: Optional identifier grouping messages in a thread.
            parent_id: Optional identifier of the direct parent post.
            history_records: List of historical record dictionaries with keys:
                'id', 'text', 'author', 'target_user', 'timestamp', 'thread_id', 'parent_id'

        Returns:
            Dict containing:
                - parent_text: str (text of parent message if known)
                - previous_messages: List[str] (up to 3 preceding messages)
                - turn_count: int (number of turns in the conversation prior to this message)
                - time_delta_seconds: float (seconds since the most recent preceding message)
                - context_snippet: str (formatted preceding context for transformer input)
        """
        if not history_records:
            return {
                "parent_text": "",
                "previous_messages": [],
                "turn_count": 0,
                "time_delta_seconds": 0.0,
                "context_snippet": ""
            }

        # Filter records strictly strictly occurring BEFORE the current timestamp (NO LEAKAGE)
        prior_records = []
        for r in history_records:
            r_ts = r.get("timestamp")
            if isinstance(r_ts, str):
                try:
                    r_ts = datetime.fromisoformat(r_ts)
                except Exception:
                    continue
            if r_ts and r_ts < timestamp:
                prior_records.append({**r, "parsed_ts": r_ts})

        # Match relevant conversation: either by thread_id or between author/target pair
        relevant_records = []
        parent_text = ""

        # Check for direct parent first
        if parent_id:
            for r in prior_records:
                if str(r.get("id")) == str(parent_id):
                    parent_text = str(r.get("text", ""))
                    break

        if thread_id:
            relevant_records = [
                r for r in prior_records 
                if r.get("thread_id") == thread_id
            ]
        
        # If thread_id produced nothing or wasn't provided, look for direct exchanges between author and target
        if not relevant_records and target_user and target_user != "@anonymous":
            relevant_records = [
                r for r in prior_records
                if (r.get("author") in (author, target_user) and r.get("target_user") in (author, target_user))
            ]

        # Sort chronologically
        relevant_records.sort(key=lambda x: x["parsed_ts"])

        # If parent_text not resolved via parent_id, take the last message before this one as parent
        if not parent_text and relevant_records:
            parent_text = str(relevant_records[-1].get("text", ""))

        # Take up to the 3 most recent preceding messages
        recent_3 = relevant_records[-3:] if relevant_records else []
        previous_messages = [str(r.get("text", "")) for r in recent_3]

        turn_count = len(relevant_records)

        # Time delta in seconds since the last message
        time_delta_seconds = 0.0
        if relevant_records:
            last_ts = relevant_records[-1]["parsed_ts"]
            time_delta_seconds = max(0.0, (timestamp - last_ts).total_seconds())

        # Construct context snippet for transformer input:
        # e.g. "Previous: [msg1] | [msg2] | Parent: [parent_text]"
        snippet_parts = []
        if parent_text:
            snippet_parts.append(f"Parent: {parent_text}")
        elif previous_messages:
            snippet_parts.append("Preceding: " + " | ".join(previous_messages))

        context_snippet = " ".join(snippet_parts)

        return {
            "parent_text": parent_text,
            "previous_messages": previous_messages,
            "turn_count": turn_count,
            "time_delta_seconds": round(time_delta_seconds, 2),
            "context_snippet": context_snippet
        }
