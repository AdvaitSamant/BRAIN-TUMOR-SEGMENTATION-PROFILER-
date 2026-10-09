"""Primary Streamlit entry point for the trained MONAI tumor model."""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).resolve().with_name("monai_app.py")), run_name="__main__")

# Trigger reload

# Trigger reload 2
