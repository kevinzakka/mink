:github_url: https://github.com/kevinzakka/mink/tree/main/docs/installation.rst

Installation
============

``mink`` is distributed on PyPI and supports Python 3.10 and above:

.. tab-set::

   .. tab-item:: uv

      .. code:: bash

         uv add mink

   .. tab-item:: pip

      .. code:: bash

         pip install mink

Verification
------------

.. tab-set::

   .. tab-item:: uv

      .. code:: bash

         uv run python -c "from mink import Configuration; print('OK')"

   .. tab-item:: pip

      .. code:: bash

         python -c "from mink import Configuration; print('OK')"

Development Installation
------------------------

Clone the repository and install all dependencies:

.. code:: bash

   git clone https://github.com/kevinzakka/mink.git && cd mink
   uv sync --all-groups

Common development commands:

.. code:: bash

   make test      # Run tests
   make check     # Run formatter (ruff) and type checkers (ty, pyright)
   make doc       # Build documentation
   make doc-live  # Build docs with live reload

See `CONTRIBUTING.md <https://github.com/kevinzakka/mink/blob/main/CONTRIBUTING.md>`_ for guidelines.
