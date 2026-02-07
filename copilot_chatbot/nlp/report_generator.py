"""\
SmartShelf AI - Report Generator

Generate narrative summaries and business reports from structured data context.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate insight narratives in a consistent structure."""

    def generate_insight_narrative(
        self,
        data_context: Dict[str, Any],
        narrative_type: str = "summary",
        tone: str = "professional",
    ) -> Dict[str, Any]:
        # Keep deterministic template-based generation for offline demo.
        title = data_context.get("title") or "Automated Business Report"
        period = data_context.get("period") or "recent period"
        highlights = data_context.get("highlights") or []
        metrics = data_context.get("metrics") or {}
        recommendations = data_context.get("recommendations") or []

        tone_prefix = ""
        if tone == "urgent":
            tone_prefix = "Immediate attention needed. "
        elif tone == "casual":
            tone_prefix = "Here's a quick update. "

        lines = []
        lines.append(f"{tone_prefix}{title} for {period}.")

        if metrics:
            mparts = []
            for k, v in metrics.items():
                mparts.append(f"{k}: {v}")
            lines.append("Key metrics: " + ", ".join(mparts) + ".")

        if highlights:
            lines.append("Key findings: " + "; ".join(highlights) + ".")

        if narrative_type == "recommendation" and recommendations:
            lines.append("Recommended actions: " + "; ".join(recommendations) + ".")
        elif recommendations:
            lines.append("Next steps: " + "; ".join(recommendations) + ".")

        generated = " ".join(lines)
        return {"generated_text": generated}
