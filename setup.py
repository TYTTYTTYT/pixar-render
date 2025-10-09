import subprocess
import sys
import os
import tomllib
from setuptools import setup, find_packages
from typing import List, Tuple, Union, Dict
from types import EllipsisType

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
                "resources/environment.yml",
            ],
        },
        include_package_data=True,
    )

if __name__ == "__main__":
    main()
