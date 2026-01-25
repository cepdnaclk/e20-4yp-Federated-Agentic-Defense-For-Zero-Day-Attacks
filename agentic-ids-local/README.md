# Project Documentation

## Overview
Make a  quick introduction about this project.

## File Structure
- `pyproject.toml`: Configuration file defining project metadata, dependencies, and build settings.
- `src/`: Directory containing the main source code modules.
- `requirements.txt`: List of dependencies for fallback installation.
- `.venv/`: Virtual environment directory (created during setup).
- `README.md`: This documentation file.
- Other files: Include any additional scripts, configuration files (e.g., `.gitignore`), or assets as needed.

## Setup Instructions
Follow these steps to set up the project on your local machine:

### Prerequisites
- Python 3.12 or higher installed on your system.
- `uv` installed globally. If not, install it via `pip install uv` or follow the official installation guide at [uv documentation](https://github.com/astral-sh/uv).

### Steps
1. **Clone or Download the Repository**:
    - Clone the repo: `git clone https://github.com/cepdnaclk/e20-4yp-Federated-Agentic-Defense-For-Zero-Day-Attacks.git`
    - Navigate to the project directory: `cd <project-directory>`

2. **Create a Virtual Environment with uv**:
    - Run: `uv venv`
    - This creates a `.venv` directory in the project root.

3. **Activate the Virtual Environment**:
    - On Unix-like systems (Linux/macOS): `source .venv/bin/activate`
    - On Windows: `.venv\Scripts\activate`
    - Your shell prompt should now indicate the active venv (e.g., `(.venv)`).

4. **Install Dependencies**:
    - Run: `uv pip install -r requirements.txt`
    - This installs the project in editable mode along with all dependencies specified in `pyproject.toml`.


### Deactivation
- To deactivate the venv, simply run: `deactivate`

## Running the Agents in direct

To run the `directAgent.run.py` script, follow these steps:

1. Navigate to the project directory: `cd agentic-ids`
2. Create a virtual environment: `uv venv`
3. Activate the virtual environment (on Windows): `.venv\Scripts\activate`
4. Install dependencies: `uv pip install -r requirements.txt`
5. Run the script: `python directAgent.run.py`

## Important Points
- **Environment Isolation**: Always work within the activated venv to avoid conflicts with system Python packages.
- **Dependency Management**: Use `uv` for all package operations (e.g., `uv add <package>` to add new deps). Avoid mixing with `pip` outside of `uv`.
- **Python Version**: Ensure compatibility with the version specified in `pyproject.toml` (e.g., `requires-python = ">=3.12"`).
- **Security**: Regularly update dependencies with `uv lock --upgrade`.
- **Troubleshooting**: If issues arise, delete `.venv` and recreate it. Check `uv --version` to ensure it's up-to-date.


