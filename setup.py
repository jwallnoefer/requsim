from setuptools import setup

# with open("README.md", "r", encoding="utf-8") as readme:
#     long_description = readme.read()

setup(
    name="requsim",
    package_dir={"": "src"},
    packages=["requsim"],
    python_requires=">=3.8.10",
    use_scm_version={
        "write_to": "src/requsim/version.py",
        "local_scheme": "no-local-version",
    },
    setup_requires=["setuptools_scm"],
)
