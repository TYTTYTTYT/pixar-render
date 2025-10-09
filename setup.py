import subprocess
import sys
import os
import tomllib
from setuptools import setup, find_packages
from typing import List, Tuple, Union, Dict
from types import EllipsisType
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py

class build_py(_build_py):
    def run(self):
        # your custom step happens during wheel build
        check_conda_installed()
        install_conda_dependencies()
        super().run()

def check_conda_installed() -> None:
    """Check if Conda is installed."""
    try:
        subprocess.check_call(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Conda is installed!")
    except FileNotFoundError:
        print("Error: Conda is not installed. Please install Conda to proceed.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: Conda command failed. Make sure Conda is installed correctly.")
        sys.exit(1)

def install_conda_dependencies() -> None:
    """Install Conda dependencies from environment.yml."""
    path = Path(__file__).parent
    if not (path / "environment.yml").exists():
        print("Error: environment.yml not found. Please ensure the file exists in the project directory.")
        sys.exit(1)

    print("Installing Conda dependencies from environment.yml...")
    try:
        subprocess.check_call(["conda", "env", "update", "-f", "environment.yml"])
    except subprocess.CalledProcessError:
        print("Error: Conda dependencies installation failed.")
        sys.exit(1)

def read_metadata_from_pyproject() -> Tuple[str, str, List[str], Union[str, EllipsisType], Union[str, EllipsisType], Union[str, EllipsisType], Union[str, EllipsisType], List[str], Dict[str, str], str]:
    """Read project metadata (name, version, author, email, description, dependencies) from pyproject.toml using tomllib (Python 3.11+)."""
    try:
        # Open pyproject.toml in binary mode ('rb') for tomllib
        with open("pyproject.toml", "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # Extract project metadata from the [project] section
        project_metadata = pyproject_data.get("project", {})
        assert isinstance(project_metadata, dict)
        name = project_metadata.get("name", ...)
        assert isinstance(name, (str))
        version = project_metadata.get("version", ...)
        assert isinstance(version, (str))
        authors = project_metadata.get("authors", [])
        author = authors[0]['name'] if authors else ...
        assert isinstance(author, (str, EllipsisType))
        email = authors[0]['email'] if authors and 'email' in authors[0] else ...
        assert isinstance(email, (str, EllipsisType))
        description = project_metadata.get("description", ...)
        assert isinstance(description, (str, EllipsisType))

        # Read long description from README if specified
        long_description: Union[str, EllipsisType] = ...
        if "readme" in project_metadata:
            readme_path = project_metadata["readme"]
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    long_description = f.read()

        # Extract dependencies from the [project.dependencies] section
        dependencies: List[str] = project_metadata.get("dependencies", [])
        install_requires = dependencies


        classifiers = project_metadata.get('classifiers', ...)
        assert isinstance(classifiers, list)

        project_urls = project_metadata.get('urls', {})
        python_requires = project_metadata.get('requires-python', '')
        assert isinstance(python_requires, str)

        return name, version, install_requires, author, email, description, long_description, classifiers, project_urls, python_requires
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        sys.exit(1)

def main() -> None:
    """Run all installation steps."""
    # Step 1: Check if Conda is installed
    check_conda_installed()

    # Step 2: Install Conda dependencies
    install_conda_dependencies()

    # Step 3: Read project metadata (name, version, dependencies) from pyproject.toml
    name, version, install_requires, author, email, description, long_description, classifiers, urls, python_requires = read_metadata_from_pyproject()

    # Step 4: Run setup with the standard setup process
    setup(
        name=name,
        version=version,
        install_requires=install_requires,  # Install dependencies listed in pyproject.toml
        author=author, # type: ignore
        author_email=email, # type: ignore
        description=description, # type: ignore
        long_description=long_description, # type: ignore
        long_description_content_type="text/markdown" if isinstance(long_description, str) else ... , # Can be changed based on your format (e.g., reStructuredText) # type: ignore
        project_urls=urls,  # Modify this to your package's URL
        classifiers=classifiers, # type: ignore
        python_requires=python_requires,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        package_data={
            "pixar_render": [
                "resources/render_default_conf.json",
                "resources/fonts/*.ttf",
            ],
        },
        include_package_data=True,
        cmdclass={"build_py": build_py},
    )

if __name__ == "__main__":
    main()
