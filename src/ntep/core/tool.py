from dataclasses import dataclass

@dataclass
class Tool:
    """A simple dataclass representing a tool in the NTEP framework."""
    id: str
    name: str
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = self.name
