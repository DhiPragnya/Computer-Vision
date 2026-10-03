"""Shared building blocks used by every topic of the Computer Vision course.

core.theme       - the colour language (what "before", "after", e1, e2 look like)
core.live        - live sliders (update while dragging) and @live.section partial reruns
core.anim        - step-by-step derivation players and scrubbers (Plotly animations computed in Python)
core.fastplotly  - `with fast():` skips Plotly trace validation in hot paths
core.plotting    - Plotly helpers (transform figures, layers, arrows, grids)
core.ui          - page header, callouts, matrices, quizzes, lesson navigation, data-URI images
core.transforms  - topic 1 maths: matrices, homogeneous coordinates, classification, DLT, fitting, RANSAC
core.shapes      - small test shapes (house, letter F, grid, circle, ...)
core.images      - sample images + image warping (forward / inverse mapping, interpolation)
core.edges       - topic 2 maths: kernels, gradients, LoG, zero crossings, NMS, hysteresis, Canny, F1
core.edge_ui     - widgets shared by the edge lessons (image picker, kernel grids, signal plots)
"""
