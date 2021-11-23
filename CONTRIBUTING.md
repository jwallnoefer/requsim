# Contributing to requsim

Code Contributions to requsim should be made on on their own Git
branches via a pull request to its `dev` branch. The pull request will
first be reviewed and merged to `dev` and, after further evaluation,
propagated to the `main` branch from where it can be included into
an official release.


## Development Environment

This project uses `pipenv` for setting up a stable development environment.
The following assumes Python 3.8 and `pipenv` are installed on your system.
You can set up the development environment like this:

```
git clone https://github.com/jwallnoefer/requsim.git
cd requsim
git checkout dev
pipenv sync --dev
pipenv run pre-commit install
pipenv shell
```

This clones the repository and installs the necessary tools and dependencies
into a Python virtual environment, that is managed by `pipenv`. `pipenv shell`
launches a new subshell with this virtual environment activated.

## Tests

It is strongly encouraged to write tests and run them locally prior to any
pull request to the GitHub repository.
Tests are located in the `./tests/` directory and we use
[`pytest`](https://docs.pytest.org/en/stable/) as our testrunner. (Note that
some old tests are still written as `unittest` test cases.)

You can invoke `pytest` in the development environment by simply running:
```
pytest
```
This should automatically discover all the tests and run in under 10 seconds.
