from setuptools import setup, find_packages

setup(
    name="habitat-windsurf",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Will be read from requirements/base.txt
    ],
    extras_require={
        "rag": [],  # Will be read from requirements/rag.txt
        "test": [],  # Will be read from requirements/test.txt
        "dev": [],   # Will be read from requirements/dev.txt
    },
    python_requires=">=3.11",
    author="Habitat Team",
    description="Visualization components for the Habitat ecosystem",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)
