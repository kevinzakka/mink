# Contributing

Thanks for your interest in contributing to **mink**! Here are a few great ways
to get involved:

- Try out the [examples](examples) and report any issues.
- Pick something you'd like to do with one of the many [MuJoCo Menagerie
  models](https://github.com/google-deepmind/mujoco_menagerie) and
  write a new example.
- Find a use case that isn’t covered yet and write a unit test for it.
- Improve the documentation.
- Implement new tasks or constraints.

If any of those sound interesting, open an
[issue](https://github.com/kevinzakka/mink/issues) and let us know you're on it!

## Pull Requests

When submitting a pull request, please make sure to:

- **Update the [changelog](CHANGELOG.md)** with a short description of your change.
- **Run all tests** locally to ensure nothing is broken. You can do this with:

  ```bash
  pytest .
  ```

If you’re adding new functionality, consider adding corresponding tests and
updating the docs if applicable.

## Documentation

If you’re adding new functionality to mink and want to update the documentation,
you can build it and preview it locally like this:

```bash
uv pip install -r docs/requirements.txt
sphinx-build docs _build -W
open _build/index.html
```

Thanks again for helping make `mink` better!
