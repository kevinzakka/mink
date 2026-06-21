"""Execute the runnable code examples embedded in the documentation.

Each fixture under ``examples/docs/`` is the single source of truth for a
``literalinclude`` block in the docs. Running them here ensures the documented
code stays in sync with the public API.
"""

import subprocess
import sys
from pathlib import Path

from absl.testing import absltest, parameterized

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_DOC_EXAMPLES = sorted((_EXAMPLES_DIR / "docs").glob("*.py"))


class TestDocExamples(parameterized.TestCase):
    """Runs each documentation example as a standalone script."""

    @parameterized.parameters(*[(p.name,) for p in _DOC_EXAMPLES])
    def test_doc_example_runs(self, name: str):
        # Relative model paths in the examples resolve from the examples directory.
        result = subprocess.run(
            [sys.executable, f"docs/{name}"],
            check=False,
            cwd=_EXAMPLES_DIR,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"docs/{name} failed:\n{result.stdout}\n{result.stderr}",
        )


if __name__ == "__main__":
    absltest.main()
