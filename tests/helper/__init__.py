"""
tests.helper — Internal helpers for the tests package.

  visual_layer_sweep_report
      HTML report and console summary for the visual model layer sweep.
      Imported by tests.visual_layer_sweep; not a standalone CLI.
"""

from .visual_layer_sweep_report import generate_html_report, print_console_summary

__all__ = ['generate_html_report', 'print_console_summary']
