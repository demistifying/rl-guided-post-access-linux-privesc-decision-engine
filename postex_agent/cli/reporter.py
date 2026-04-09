import json
import os
from typing import Any, Dict, List

class EngagementReporter:
    """Generates human-readable markdown reports from raw JSONL agent execution logs."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def generate(self) -> str:
        if not os.path.exists(self.log_path):
            return "> _No execution log found._"

        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                events.append(json.loads(line))

        if not events:
            return "> _Empty execution log._"

        final_state = events[-1]["state_after"]
        initial_state = events[0]["state_after"]

        md = [
            "# Post-Exploitation Engagement Report\n",
            "## Target Summary",
            f"- **OS / Distribution**: {final_state.get('os_info') or 'Unknown'}",
            f"- **Kernel Version**: {final_state.get('kernel_version') or 'Unknown'}",
            f"- **Containerized**: {'Yes' if final_state.get('is_containerized') else 'No'}",
            f"- **Initial User**: {initial_state.get('current_user') or 'Unknown'}",
            f"- **Final Privilege**: **{'ROOT' if final_state.get('current_privilege') == 1 else 'User'}**",
            f"- **Total Steps**: {len(events)}\n",
        ]

        md.append("## Identified Vulnerabilities\n")
        found_any = False
        
        found_map = final_state.get("found", {})
        if found_map.get("sudo"):
            md.append("### Sudo Privileges")
            for entry in final_state.get("sudo_nopasswd_entries", []):
                md.append(f"- NOPASSWD: `{entry}`")
            found_any = True
            
        if found_map.get("suid"):
            md.append("### SUID Binaries")
            for entry in final_state.get("suid_exploitable_bins", []):
                md.append(f"- `{entry}`")
            found_any = True
            
        if found_map.get("capabilities"):
            md.append("### Capabilities")
            for entry in final_state.get("capabilities_exploitable", []):
                md.append(f"- `{entry}`")
            found_any = True
            
        if found_map.get("cron"):
            md.append("### Writable Cron Jobs")
            for entry in final_state.get("cron_writable_targets", []):
                md.append(f"- `{entry}`")
            found_any = True
            
        if found_map.get("credentials"):
            md.append("### Credentials Discovered")
            for entry in final_state.get("credentials", []):
                md.append(f"- `{str(entry)[:100]}`")
            found_any = True
            
        if final_state.get("kernel_version"):
            # If any known CVEs were in the parsed log, we could extract them from events.
            pass

        if not found_any:
            md.append("> _No significant vulnerabilities were enumerated._\n")

        md.append("\n## Attack Chain Execution Log\n")
        
        for step_data in events:
            step_num = step_data["step"]
            action = step_data["action"]
            md.append(f"### Step {step_num}: {action}")
            
            commands = step_data.get("commands", [])
            if commands:
                md.append("```bash")
                for cmd in commands:
                    md.append(f"$ {cmd}")
                md.append("```")
                
            parsed = step_data.get("parsed", {})
            if parsed.get("vector_found"):
                md.append("> **Result:** Vulnerability Confirmed ✅")
            elif "UserParser" in str(parsed) or "OSParser" in str(parsed): # generic heuristic
                md.append("> **Result:** Recon Data Extracted ℹ️")

        return "\n".join(md)
