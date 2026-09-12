"""Sphinx configuration for the Singapore AI Day DLI course site."""

import os


project = "Build High-Accuracy Vision AI Agents for Anomaly Detection"
author = "NVIDIA"
copyright = "2026 NVIDIA Corporation"

# The extension loads MyST, Sphinx Design, Copybutton, and Intersphinx.
extensions = ["sphinx_nvlearning"]

# Environment defaults propagate into the sphinx-llm Markdown subprocess.
# Keep matching -D values in release commands so the parent HTML build and its
# Markdown twins use the same edition.
nvlearning_mode = os.environ.get("NVLEARNING_MODE", "course")
nvlearning_audience = os.environ.get("NVLEARNING_AUDIENCE", "public")
nvlearning_metadata = {
    "slug": "singapore-ai-day-dli",
    "survey_source": "singapore-ai-day-dli",
    "duration": "Two parts",
    "level": "All levels",
    "delivery": "self-paced",
}

# Override any subset to demonstrate feature control. These values match the
# extension defaults and make the complete contract visible in this demo.
nvlearning_features = {
    "analytics": True,
    "standard_icon_links": True,
    "youtube": True,
    "discord": True,
    "survey": True,
    "home": True,
    "tasklists": True,
    "sidebar_resources": True,
    "ai_discovery": True,
    "view_as_markdown": True,
    "external_links_new_tab": True,
}
nvlearning_search = {"provider": "sphinx", "scripts": None}
nvlearning_nav_links = []
nvlearning_validate = True

# Set this to the final published URL before release. It provides absolute
# links in llms.txt and the generated Markdown twins.
html_baseurl = (
    "https://docs.nvidia.com/learning/physical-ai/"
    "singapore-ai-day-dli/latest/"
)

exclude_patterns = ["_includes/**"]
myst_number_code_blocks = ["python", "py", "bash", "json", "yaml"]