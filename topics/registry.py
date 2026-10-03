"""The course table of contents.

To add a new topic:
  1. create a folder  topics/tNN_<name>/  with one .py file per lesson page
  2. add an entry to TOPICS below (id, number, title, summary, folder, pages)
The sidebar, the home page cards and the previous/next buttons all read from here.

Icons use Google Material Symbols names, written as ":material/<name>:".
"""

TOPICS = [
    dict(
        id="t01",
        number=1,
        title="2D Geometric Transformations",
        summary="How points, shapes and images move - from a simple shift to a full perspective "
                "homography - and why homogeneous coordinates make it all one matrix multiply.",
        folder="topics/t01_geometric_transforms_2d",
        pages=[
            dict(file="p01_overview.py", title="The big picture", icon=":material/account_tree:"),
            dict(file="p02_homogeneous.py", title="Homogeneous coordinates", icon=":material/3d_rotation:"),
            dict(file="p03_translation.py", title="Translation", icon=":material/open_with:"),
            dict(file="p04_scaling.py", title="Scaling", icon=":material/aspect_ratio:"),
            dict(file="p05_rigid.py", title="Rotation & rigid motion", icon=":material/rotate_right:"),
            dict(file="p06_similarity.py", title="Similarity", icon=":material/zoom_in:"),
            dict(file="p07_affine.py", title="Affine", icon=":material/transform:"),
            dict(file="p08_projective.py", title="Projective (homography)", icon=":material/view_in_ar:"),
            dict(file="p09_composition.py", title="Composition & inverses", icon=":material/layers:"),
            dict(file="p10_warping.py", title="Warping images", icon=":material/image:"),
            dict(file="p11_estimation.py", title="Estimating from matches", icon=":material/scatter_plot:"),
            dict(file="p12_playground.py", title="Matrix playground", icon=":material/grid_on:"),
            dict(file="p13_recap_quiz.py", title="Recap & quiz", icon=":material/quiz:"),
        ],
    ),
    dict(
        id="t02",
        number=2,
        title="Edge Detection",
        summary="Where does intensity change, and which changes matter? From the derivative of a step, "
                "through Sobel, the Laplacian of Gaussian and Canny, to learned multi-scale detectors.",
        folder="topics/t02_edge_detection",
        pages=[
            dict(file="p01_what_is_an_edge.py", title="What is an edge?", icon=":material/show_chart:"),
            dict(file="p02_derivatives.py", title="Derivatives on pixels", icon=":material/functions:"),
            dict(file="p03_sobel_prewitt.py", title="Sobel & Prewitt", icon=":material/blur_linear:"),
            dict(file="p04_laplacian.py", title="Second derivative & Laplacian", icon=":material/swap_vert:"),
            dict(file="p05_gaussian_log.py", title="Gaussian & LoG", icon=":material/blur_on:"),
            dict(file="p06_canny_nms.py", title="Canny I: non-maximum suppression", icon=":material/line_weight:"),
            dict(file="p07_canny_hysteresis.py", title="Canny II: thresholds & hysteresis", icon=":material/timeline:"),
            dict(file="p08_canny_complete.py", title="Canny III: why it works & comparison", icon=":material/tune:"),
            dict(file="p09_learned_edges.py", title="Learned edge detectors", icon=":material/neurology:"),
            dict(file="p10_recap_quiz.py", title="Recap & quiz", icon=":material/quiz:"),
        ],
    ),
    dict(
        id="t03",
        number=3,
        title="Contours, Polygons & Hough",
        summary="From edge pixels to shapes: trace ordered contours, measure and simplify them (shoelace, RDP, "
                "circularity), and find lines and circles by voting in parameter space - even across gaps.",
        folder="topics/t03_contours_hough",
        pages=[
            dict(file="p01_edges_to_contours.py", title="From edges to contours", icon=":material/gesture:"),
            dict(file="p02_connectivity.py", title="Connectivity & boundary pixels", icon=":material/grid_4x4:"),
            dict(file="p03_tracing.py", title="Contour tracing & hierarchy", icon=":material/route:"),
            dict(file="p04_length_area.py", title="Contour length & area", icon=":material/square_foot:"),
            dict(file="p05_rdp.py", title="Polygonal approximation (RDP)", icon=":material/polyline:"),
            dict(file="p06_measurements.py", title="ε trade-off & shape measurements", icon=":material/straighten:"),
            dict(file="p07_hough_mc.py", title="Hough idea: voting for (m, c)", icon=":material/how_to_vote:"),
            dict(file="p08_hough_lines.py", title="Hough lines in (ρ, θ)", icon=":material/show_chart:"),
            dict(file="p09_hough_circles.py", title="Hough circles", icon=":material/radio_button_unchecked:"),
            dict(file="p10_hough_general.py", title="General Hough, gaps & limits", icon=":material/hub:"),
            dict(file="p11_recap_quiz.py", title="Recap & quiz", icon=":material/quiz:"),
        ],
    ),
    # Next topics go here, e.g.
    # dict(id="t04", number=4, title="...", summary="...", folder="topics/t04_...", pages=[...]),
]


def page_path(topic: dict, page: dict) -> str:
    return f"{topic['folder']}/{page['file']}"


def url_path(topic: dict, page: dict) -> str:
    stem = page["file"].rsplit(".", 1)[0].split("_", 1)[-1]
    return f"{topic['id']}-{stem}"


def locate(file: str):
    """Find (topic, index, page) for a page script given its __file__."""
    from pathlib import Path

    p = Path(file)
    for topic in TOPICS:
        for i, page in enumerate(topic["pages"]):
            if p.name == page["file"] and p.parent.name == Path(topic["folder"]).name:
                return topic, i, page
    return None, None, None
