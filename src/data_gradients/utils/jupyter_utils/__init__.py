""" Block notebook cells from running while interacting with widgets
https://github.com/Kirill888/jupyter-ui-poll/blob/develop/jupyter_ui_poll/__init__.py
"""

from ._poll import ui_events, with_ui_events, run_ui_poll_loop

__all__ = (
    "ui_events",
    "with_ui_events",
    "run_ui_poll_loop",
)
