from setuptools import setup, find_packages

# with open("README.md", "r", encoding="utf-8") as readme:
#     long_description = readme.read()

setup(
    name="requsim",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8.10",
    install_requires=["numpy>=1.21.2", "pandas>=1.3.3"],
    extras_require={
        "test": ["pytest>=6.2.5"],
        "docs": ["sphinx", "autodocsumm", "recommonmark", "sphinx-rtd-theme"],
    },
)
