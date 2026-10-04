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

Development dependencies require Python 3.10.12 or later.

Tests load robot models with the pinned ``mujoco-menagerie`` package. Examples
keep their local XML files for custom scenes, actuators, and robot combinations.
They use packaged meshes and textures unless a custom asset remains local.

The first use of a packaged model downloads its assets and requires network
access. The package stores each model in a per-user cache. Later runs reuse
that cache and can run without network access. Set ``MENAGERIE_CACHE_DIR`` to
select a different cache directory.

Common development commands:

.. code:: bash

   make test      # Run tests
   make check     # Run formatter (ruff) and type checkers (ty, pyright)
   make doc       # Build documentation
   make doc-live  # Build docs with live reload

The normal test command uses the versions in ``uv.lock``. Run
``make test-latest`` to test the current code with the latest compatible runtime
dependencies. This command uses an isolated environment and does not update
``uv.lock`` or the project environment.

See `CONTRIBUTING.md <https://github.com/kevinzakka/mink/blob/main/CONTRIBUTING.md>`_ for guidelines.
