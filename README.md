# Analytics to Action Exam Project

## Getting Started

### Environment Setup

1. Install `uv` (recommended over regular venv for faster dependency resolution):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix-like systems
# OR
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Data Requirements

**Note:** Due to data privacy and size constraints, the dataset is not included in this repository. To run this project, you will need to:

1. Obtain the required dataset from the course materials
2. Place the data files in the `data/` directory
3. Ensure your data files match the expected format as described in the notebooks

## Project Structure

- `notebooks/`: Jupyter notebooks containing analysis
- `data/`: Place your dataset files here (not included in repo)
- `src/`: Source code for reusable functions
- `requirements.txt`: Project dependencies

## Contributing

### Before You Start
1. Make sure you have the latest version of the main branch:
   ```bash
   git checkout main
   git pull origin main
   ```

### Making Your Changes
1. Create a new branch with a descriptive name:
   ```bash
   # Replace feature-name with something descriptive like "add-data-visualization"
   git checkout -b feature-name
   ```

2. Make your changes to the code
   - Edit files
   - Add new files
   - Test your changes

3. Save your changes to Git:
   ```bash
   # See which files you've changed
   git status

   # Add all changed files
   git add .
   
   # Or add specific files
   git add filename.py

   # Create a commit with a clear message
   git commit -m "Add clear description of your changes"
   ```

4. Upload your changes:
   ```bash
   git push origin feature-name
   ```
   If this is your first push, Git will show you a link to create a Pull Request.

### Creating a Pull Request (PR)
1. Go to the repository on GitHub
2. You should see a yellow banner suggesting to create a Pull Request - click it
3. If not, click the "Pull Requests" tab and then the green "New Pull Request" button

### In the Pull Request
1. Set the "base" branch to `main`
2. Set the "compare" branch to your branch (`feature-name`)
3. Fill in the PR template:
   - Title: Brief description of what you did
   - Description: 
     - What changes did you make?
     - Why did you make these changes?
     - Any special instructions for testing?

### After Creating the PR
1. Wait for review
2. Make any requested changes:
   ```bash
   # Make your changes
   git add .
   git commit -m "Address review comments"
   git push origin feature-name
   ```
3. Once approved and merged, clean up:
   ```bash
   # Switch back to main
   git checkout main
   
   # Get the latest changes
   git pull origin main
   
   # Delete your local branch
   git branch -d feature-name
   ```

### Tips
- Commit often with clear messages
- One feature per branch
- Ask for help if you get stuck!
- Keep your PRs small and focused

## License

[Your license information here]

