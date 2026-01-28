from typing import List, Dict, Set

class DependencyGraph:
    """Manages the partial order (≤) of tool dependencies."""
    def __init__(self, dependencies: Dict[str, List[str]]):
        """
        dependencies: A dict where key is a tool_id and value is a list of
                      tool_ids that must come *before* the key.
                      e.g., {"package": ["resize", "encrypt"]}
        """
        self.dependencies = dependencies
        self._build_prerequisite_map()

    def _build_prerequisite_map(self):
        """Creates a map of what each tool is a prerequisite for."""
        self.prerequisite_map = {tool_id: [] for tool_id in self.dependencies}
        for tool, prereqs in self.dependencies.items():
            for prereq in prereqs:
                self.prerequisite_map.setdefault(prereq, []).append(tool)

    def is_valid_next_step(self, tool_id: str, executed_tools: Set[str]) -> bool:
        """Checks if a tool can be executed next given the history."""
        # All its prerequisites must be in the executed_tools set
        return all(prereq in executed_tools for prereq in self.dependencies.get(tool_id, []))

    def is_valid_chain(self, chain: List[str]) -> bool:
        """Validates an entire chain of tool IDs."""
        executed = set()
        for tool_id in chain:
            if not self.is_valid_next_step(tool_id, executed):
                return False
            executed.add(tool_id)
        return True
