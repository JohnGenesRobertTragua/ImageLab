import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk, ImageFilter, ImageOps, ExifTags
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageLabApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "ImageLab - Image Processing & Enhancement System"
        )

        # Start maximized so ImageLab opens using the available screen space,
        # while remaining a normal resizable/maximizable window.
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)
        self.root.resizable(True, True)
        self.root.after(0, self.root.state, "zoomed")

        # ========================================================
        # COLORS
        # ========================================================

        self.bg_color = "#eef2f7"
        self.sidebar_color = "#172033"
        self.sidebar_hover = "#263650"
        self.sidebar_active = "#2f6fed"
        self.card_color = "#ffffff"
        self.text_color = "#172033"
        self.secondary_text = "#64748b"
        self.accent_color = "#2563eb"
        self.accent_hover = "#1d4ed8"
        self.border_color = "#d7dee8"
        self.soft_accent = "#eaf1ff"
        self.success_color = "#16a34a"

        # ========================================================
        # IMAGE STATE
        # ========================================================

        self.original_image = None
        self.base_image = None
        self.current_image = None

        # ========================================================
        # LIVE PREVIEW / PERFORMANCE STATE
        # ========================================================

        # Large images are processed at a reduced resolution while
        # a slider is being dragged. Full resolution is restored
        # when the edit is finished or when the image is saved.
        self.preview_max_dimension = 900
        self._preview_base_image = None
        self._preview_source_id = None
        self.live_preview_after_id = None
        self.is_slider_dragging = False

        self.display_image = None
        self.image_path = None

        # ========================================================
        # PROCESSING STATE
        # ========================================================

        self.brightness_value = 0
        self.contrast_value = 0
        self.current_filter = "Original"
        self.filter_strength = 3.0
        self.filter_strengths = {
            "Blur": 3.0,
            "Sharpen": 2.0,
            "Smooth": 3.0
        }

        # ========================================================
        # ADVANCED PROCESSING STATE
        # ========================================================

        self.saturation_value = 0
        self.hue_value = 0
        self.red_value = 0
        self.green_value = 0
        self.blue_value = 0
        self.gamma_value = 1.0
        self.threshold_value = 128
        self.advanced_mode = "None"
        self.active_tool = None
        self.tool_buttons = {}

        # ========================================================
        # ZOOM STATE
        # ========================================================

        self.zoom_level = 1.0
        self.zoom_mode = "fit"

        # ========================================================
        # APPLICATION SETTINGS
        # ========================================================

        self.settings_max_history = 30
        self.settings_preview_quality = 900
        self.settings_live_preview = True
        self.settings_confirm_reset = True
        self.settings_default_format = "PNG"
        self.settings_jpeg_quality = 95
        self.settings_window = None

        # ========================================================
        # UNDO / REDO
        # ========================================================

        self.undo_stack = []
        self.redo_stack = []
        self.history = []

        self.is_restoring_history = False

        self.pending_slider_state = None
        self.pending_slider_type = None

        # ========================================================
        # NON-DESTRUCTIVE PROCESSING STACK
        # ========================================================

        self.processing_layers = []
        self.processing_stack_window = None
        self.processing_stack_listbox = None
        self.processing_stack_status = None
        self.pending_processing_layer_description = None

        # ========================================================
        # GUI STATE
        # ========================================================

        self.preview_label = None
        self.preview_card = None
        self.status = None

        # Inline crop editor state
        self.crop_mode = False
        self.crop_canvas = None
        self.crop_image_photo = None
        self.crop_box = None
        self.crop_scale = 1.0
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        self.crop_display_size = (0, 0)
        self.crop_rect_id = None
        self.crop_handles = {}
        self.crop_drag_data = {}
        self.crop_zoom = 1.0
        self.crop_pan_x = 0.0
        self.crop_pan_y = 0.0
        self.crop_frame = None
        self.crop_image_bounds = None

        self.brightness_frame = None
        self.contrast_frame = None
        self.filters_frame = None
        self.transform_frame = None
        self.advanced_frame = None

        self.brightness_slider = None
        self.contrast_slider = None
        self.filter_strength_slider = None
        self.filter_strength_sliders = {}
        self.filter_strength_value_labels = {}
        self.advanced_sliders = {}
        self.contrast_slider = None

        self.analysis_window = None
        self.analysis_figure = None
        self.analysis_canvas = None
        self.analysis_stats_frame = None
        self.analysis_chart_frame = None

        self.metadata_window = None
        self.metadata_content = None

        self.comparison_window = None
        self.comparison_mode = "Side by Side"
        self.comparison_zoom = 1.0
        self.comparison_canvas = None
        self.comparison_before_photo = None
        self.comparison_after_photo = None
        self.comparison_split_position = 0.5
        self.comparison_dragging_split = False
        self.comparison_before_region = None
        self.comparison_after_region = None

        # ========================================================
        # BATCH PROCESSING STATE
        # ========================================================

        self.batch_window = None
        self.batch_files = []
        self.batch_listbox = None
        self.batch_output_var = None
        self.batch_progress = None
        self.batch_status_label = None
        self.batch_process_button = None
        self.batch_brightness_var = None
        self.batch_contrast_var = None
        self.batch_saturation_var = None
        self.batch_hue_var = None
        self.batch_red_var = None
        self.batch_green_var = None
        self.batch_blue_var = None
        self.batch_gamma_var = None
        self.batch_threshold_var = None
        self.batch_filter_var = None
        self.batch_advanced_var = None
        self.batch_resize_var = None
        self.batch_format_var = None

        # ========================================================
        # COMPARISON / EVALUATION STATE
        # ========================================================

        self.evaluation_window = None
        self.evaluation_canvas = None
        self.evaluation_figure = None
        self.evaluation_report = None

        self.history_window = None

        self.history_listbox = None

        # ========================================================
        # SETUP
        # ========================================================

        self.setup_style()
        self.create_interface()
        self.setup_shortcuts()

    # ============================================================
    # STYLE
    # ============================================================

    def setup_style(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # --------------------------------------------------------
        # Base controls
        # --------------------------------------------------------
        style.configure(
            "TButton",
            font=("Segoe UI", 10),
            padding=(12, 7),
            relief="flat",
            borderwidth=0
        )

        # Main editing controls
        style.configure(
            "Tool.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 9),
            foreground="#172033",
            background="#ffffff",
            bordercolor="#d7dee8",
            lightcolor="#ffffff",
            darkcolor="#d7dee8",
            relief="solid",
            borderwidth=1
        )
        style.map(
            "Tool.TButton",
            background=[
                ("pressed", self.accent_hover),
                ("active", self.soft_accent)
            ],
            foreground=[
                ("pressed", "#ffffff"),
                ("active", self.accent_color)
            ],
            bordercolor=[
                ("active", self.accent_color),
                ("pressed", self.accent_hover)
            ]
        )

        # Active main editing tool
        style.configure(
            "ToolActive.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 9),
            foreground="#ffffff",
            background=self.accent_color,
            bordercolor=self.accent_color,
            lightcolor=self.accent_color,
            darkcolor=self.accent_color,
            relief="solid",
            borderwidth=1
        )
        style.map(
            "ToolActive.TButton",
            background=[
                ("pressed", self.accent_hover),
                ("active", self.accent_hover)
            ],
            foreground=[
                ("pressed", "#ffffff"),
                ("active", "#ffffff")
            ],
            bordercolor=[
                ("active", self.accent_hover),
                ("pressed", self.accent_hover)
            ]
        )

        # Sidebar navigation
        style.configure(
            "Sidebar.TButton",
            font=("Segoe UI", 9),
            padding=(12, 6),
            anchor="w",
            foreground="#e8eef8",
            background=self.sidebar_color,
            bordercolor=self.sidebar_color,
            lightcolor=self.sidebar_color,
            darkcolor=self.sidebar_color,
            relief="flat",
            borderwidth=0
        )
        style.map(
            "Sidebar.TButton",
            background=[
                ("pressed", self.sidebar_active),
                ("active", self.sidebar_hover)
            ],
            foreground=[
                ("pressed", "#ffffff"),
                ("active", "#ffffff")
            ]
        )

        # Small utility buttons such as zoom controls
        style.configure(
            "Utility.TButton",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            foreground="#334155",
            background="#ffffff",
            bordercolor="#d7dee8",
            lightcolor="#ffffff",
            darkcolor="#d7dee8",
            relief="solid",
            borderwidth=1
        )
        style.map(
            "Utility.TButton",
            background=[
                ("pressed", self.accent_color),
                ("active", self.soft_accent)
            ],
            foreground=[
                ("pressed", "#ffffff"),
                ("active", self.accent_color)
            ],
            bordercolor=[
                ("active", self.accent_color),
                ("pressed", self.accent_color)
            ]
        )

        # General labels / notebook-like section headings
        style.configure(
            "Section.TLabel",
            font=("Segoe UI", 10, "bold"),
            foreground=self.text_color,
            background=self.card_color
        )

    # ============================================================
    # KEYBOARD SHORTCUTS
    # ============================================================

    def setup_shortcuts(self):

        self.root.bind(
            "<Control-o>",
            lambda event: self.open_image()
        )

        self.root.bind(
            "<Control-s>",
            lambda event: self.save_image()
        )

        self.root.bind(
            "<Control-z>",
            lambda event: self.undo()
        )

        self.root.bind(
            "<Control-y>",
            lambda event: self.redo()
        )

        self.root.bind(
            "<Control-r>",
            lambda event: self.reset_image()
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.close_active_window()
        )

        self.root.bind(
            "<plus>",
            lambda event: self.zoom_in()
        )

        self.root.bind(
            "<equal>",
            lambda event: self.zoom_in()
        )

        self.root.bind(
            "<minus>",
            lambda event: self.zoom_out()
        )

    # ============================================================
    # MAIN INTERFACE
    # ============================================================

    def create_interface(self):

        # --------------------------------------------------------
        # HEADER
        # --------------------------------------------------------

        header = tk.Frame(
            self.root,
            bg=self.card_color,
            height=70
        )

        header.pack(
            side="top",
            fill="x"
        )

        header.pack_propagate(False)

        title_frame = tk.Frame(
            header,
            bg=self.card_color
        )

        # Keep the title block vertically centered inside the header.
        title_frame.place(
            relx=0.0,
            rely=0.5,
            x=25,
            anchor="w"
        )

        tk.Label(
            title_frame,
            text="ImageLab",
            font=("Segoe UI", 22, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            pady=0
        )

        tk.Label(
            title_frame,
            text="Image Processing & Enhancement System",
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            anchor="w",
            pady=0
        )

        # --------------------------------------------------------
        # MAIN CONTAINER
        # --------------------------------------------------------

        main_container = tk.Frame(
            self.root,
            bg=self.bg_color
        )

        main_container.pack(
            fill="both",
            expand=True
        )

        # --------------------------------------------------------
        # SIDEBAR
        # --------------------------------------------------------

        sidebar = tk.Frame(
            main_container,
            bg=self.sidebar_color,
            width=220
        )

        sidebar.pack(
            side="left",
            fill="y"
        )

        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="IMAGELAB",
            font=("Segoe UI", 13, "bold"),
            bg=self.sidebar_color,
            fg="#ffffff"
        ).pack(
            pady=(14, 2)
        )

        tk.Label(
            sidebar,
            text="IMAGE EDITOR",
            font=("Segoe UI", 7, "bold"),
            bg=self.sidebar_color,
            fg="#8ea3c7"
        ).pack(
            pady=(0, 10)
        )

        self.create_sidebar_button(
            sidebar,
            "Open Image",
            self.open_image
        )

        self.create_sidebar_button(
            sidebar,
            "Save Image",
            self.save_image
        )

        self.create_sidebar_button(
            sidebar,
            "Reset Image",
            self.reset_image
        )

        self.create_sidebar_button(
            sidebar,
            "Undo",
            self.undo
        )

        self.create_sidebar_button(
            sidebar,
            "Redo",
            self.redo
        )

        self.create_sidebar_button(
            sidebar,
            "Before / After",
            self.show_before_after
        )

        self.create_sidebar_button(
            sidebar,
            "Processing History",
            self.show_history
        )

        self.create_sidebar_button(
            sidebar,
            "Processing Stack",
            self.show_processing_stack
        )

        self.create_sidebar_button(
            sidebar,
            "Image Analysis",
            self.show_analysis
        )

        self.create_sidebar_button(
            sidebar,
            "Image Comparison",
            self.show_image_evaluation
        )

        self.create_sidebar_button(
            sidebar,
            "Image Information",
            self.show_metadata
        )

        self.create_sidebar_button(
            sidebar,
            "Batch Processing",
            self.show_batch_processing
        )

        self.create_sidebar_button(
            sidebar,
            "Presets",
            self.show_presets
        )

        self.create_sidebar_button(
            sidebar,
            "Settings",
            self.show_settings
        )

        # --------------------------------------------------------
        # WORKSPACE
        # --------------------------------------------------------

        workspace = tk.Frame(
            main_container,
            bg=self.bg_color
        )

        workspace.pack(
            side="left",
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        workspace.grid_rowconfigure(
            0,
            weight=1
        )

        workspace.grid_rowconfigure(
            1,
            weight=0
        )

        workspace.grid_rowconfigure(
            2,
            weight=0
        )

        workspace.grid_rowconfigure(
            3,
            weight=0
        )

        workspace.grid_columnconfigure(
            0,
            weight=1
        )

        # --------------------------------------------------------
        # PREVIEW CARD
        # --------------------------------------------------------

        self.preview_card = tk.Frame(
            workspace,
            bg=self.card_color,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.border_color
        )

        self.preview_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            pady=(0, 8)
        )

        self.preview_card.bind(
            "<Configure>",
            lambda event: self.display_image_on_preview()
        )

        preview_header = tk.Frame(
            self.preview_card,
            bg=self.card_color,
            height=44
        )

        preview_header.pack(
            fill="x"
        )

        preview_header.pack_propagate(False)

        tk.Label(
            preview_header,
            text="Image Preview",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            side="left",
            padx=(15, 4)
        )

        tk.Label(
            preview_header,
            text="WORKSPACE",
            font=("Segoe UI", 7, "bold"),
            bg=self.card_color,
            fg=self.accent_color
        ).pack(
            side="left",
            padx=(0, 8)
        )

        # --------------------------------------------------------
        # ZOOM CONTROLS
        # --------------------------------------------------------

        zoom_controls = tk.Frame(
            preview_header,
            bg=self.card_color
        )

        zoom_controls.pack(
            side="right",
            padx=10
        )

        ttk.Button(
            zoom_controls,
            text="−",
            width=3,
            command=self.zoom_out,
            style="Utility.TButton"
        ).pack(
            side="left",
            padx=2
        )

        self.zoom_label = tk.Label(
            zoom_controls,
            text="Fit",
            width=7,
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        )

        self.zoom_label.pack(
            side="left"
        )

        ttk.Button(
            zoom_controls,
            text="+",
            width=3,
            command=self.zoom_in,
            style="Utility.TButton"
        ).pack(
            side="left",
            padx=2
        )

        ttk.Button(
            zoom_controls,
            text="100%",
            command=self.zoom_100,
            style="Utility.TButton"
        ).pack(
            side="left",
            padx=2
        )

        ttk.Button(
            zoom_controls,
            text="Fit",
            command=self.zoom_fit,
            style="Utility.TButton"
        ).pack(
            side="left",
            padx=2
        )

        # --------------------------------------------------------
        # PREVIEW LABEL
        # --------------------------------------------------------

        self.preview_label = tk.Label(
            self.preview_card,
            text="No image selected\n\nClick 'Open Image' to begin",
            font=("Segoe UI", 13),
            bg=self.card_color,
            fg=self.secondary_text,
            justify="center"
        )

        self.preview_label.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(5, 10)
        )

        self.preview_label.bind(
            "<MouseWheel>",
            self.handle_mouse_zoom
        )

        self.preview_label.bind(
            "<Button-4>",
            self.handle_mouse_zoom
        )

        self.preview_label.bind(
            "<Button-5>",
            self.handle_mouse_zoom
        )

        # --------------------------------------------------------
        # TOOL BUTTONS
        # --------------------------------------------------------

        tools = tk.Frame(
            workspace,
            bg=self.bg_color
        )

        tools.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        for i in range(6):

            tools.grid_columnconfigure(
                i,
                weight=1
            )

        # Main editing tools. The selected tool stays highlighted.
        tool_definitions = [
            ("Brightness", self.show_brightness_control, "brightness"),
            ("Contrast", self.show_contrast_control, "contrast"),
            ("Filters", self.show_filters_control, "filters"),
            ("Transform", self.show_transform_control, "transform"),
            ("Analyze", self.show_analysis, "analyze"),
            ("Advanced", self.show_advanced_control, "advanced"),
        ]

        for column, (label, command, tool_id) in enumerate(tool_definitions):
            button = ttk.Button(
                tools,
                text=label,
                command=command,
                style="Tool.TButton"
            )
            button.grid(
                row=0,
                column=column,
                padx=3,
                sticky="ew"
            )
            self.tool_buttons[tool_id] = button

        # --------------------------------------------------------
        # ADJUSTMENT CONTAINER
        # --------------------------------------------------------

        self.adjustment_container = tk.Frame(
            workspace,
            bg=self.card_color,
            height=125,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.border_color
        )

        self.adjustment_container.grid(
            row=2,
            column=0,
            sticky="ew"
        )

        self.adjustment_container.grid_propagate(False)

        self.brightness_frame = tk.Frame(
            self.adjustment_container,
            bg=self.card_color
        )

        self.contrast_frame = tk.Frame(
            self.adjustment_container,
            bg=self.card_color
        )

        self.filters_frame = tk.Frame(
            self.adjustment_container,
            bg=self.card_color
        )

        self.transform_frame = tk.Frame(
            self.adjustment_container,
            bg=self.card_color
        )

        self.advanced_frame = tk.Frame(
            self.adjustment_container,
            bg=self.card_color
        )

        # --------------------------------------------------------
        # SHORTCUT INFO
        # --------------------------------------------------------

        shortcut_frame = tk.Frame(
            workspace,
            bg=self.bg_color
        )

        shortcut_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(6, 0)
        )

        tk.Label(
            shortcut_frame,
            text=(
                "Ctrl+O Open   •   "
                "Ctrl+S Save   •   "
                "Ctrl+Z Undo   •   "
                "Ctrl+Y Redo   •   "
                "Ctrl+R Reset   •   "
                "Mouse Wheel Zoom"
            ),
            font=("Segoe UI", 8),
            bg=self.bg_color,
            fg=self.secondary_text
        ).pack(
            anchor="center"
        )

        # --------------------------------------------------------
        # STATUS BAR
        # --------------------------------------------------------

        self.status = tk.Label(
            self.root,
            text="Ready - Open an image to begin",
            anchor="w",
            bg="#e2e8f0",
            fg="#334155",
            font=("Segoe UI", 9),
            padx=12,
            pady=6
        )

        self.status.pack(
            side="bottom",
            fill="x"
        )

    # ============================================================
    # SIDEBAR BUTTON
    # ============================================================

    def create_sidebar_button(
        self,
        parent,
        text,
        command
    ):

        sidebar_icons = {
            "Open Image": "📂",
            "Save Image": "💾",
            "Reset Image": "↺",
            "Undo": "↶",
            "Redo": "↷",
            "Before / After": "◐",
            "Processing History": "☷",
            "Processing Stack": "▤",
            "Image Analysis": "📊",
            "Image Comparison": "⇄",
            "Image Information": "ⓘ",
            "Batch Processing": "▦",
            "Presets": "✦",
            "Settings": "⚙",
        }

        icon = sidebar_icons.get(text, "•")

        ttk.Button(
            parent,
            text=f"{icon}  {text}",
            command=command,
            style="Sidebar.TButton"
        ).pack(
            fill="x",
            padx=10,
            pady=2
        )

    # ============================================================
    # ZOOM
    # ============================================================

    def zoom_fit(self):

        if self.current_image is None:
            return

        self.zoom_mode = "fit"
        self.zoom_level = 1.0

        self.update_zoom_label()
        self.display_image_on_preview()

        self.status.config(
            text="Zoom: Fit to window"
        )

    def zoom_100(self):

        if self.current_image is None:
            return

        self.zoom_mode = "manual"
        self.zoom_level = 1.0

        self.update_zoom_label()
        self.display_image_on_preview()

        self.status.config(
            text="Zoom: 100%"
        )

    def zoom_in(self):

        if self.current_image is None:
            return

        if self.zoom_mode == "fit":
            self.zoom_level = 1.0
            self.zoom_mode = "manual"

        self.zoom_level = min(
            5.0,
            self.zoom_level + 0.1
        )

        self.update_zoom_label()
        self.display_image_on_preview()

        self.status.config(
            text=f"Zoom: {self.zoom_level * 100:.0f}%"
        )

    def zoom_out(self):

        if self.current_image is None:
            return

        if self.zoom_mode == "fit":
            self.zoom_level = 1.0
            self.zoom_mode = "manual"

        self.zoom_level = max(
            0.1,
            self.zoom_level - 0.1
        )

        self.update_zoom_label()
        self.display_image_on_preview()

        self.status.config(
            text=f"Zoom: {self.zoom_level * 100:.0f}%"
        )

    def handle_mouse_zoom(self, event):

        if self.current_image is None:
            return

        if getattr(event, "delta", 0) > 0:

            self.zoom_in()

        elif getattr(event, "delta", 0) < 0:

            self.zoom_out()

        elif getattr(event, "num", None) == 4:

            self.zoom_in()

        elif getattr(event, "num", None) == 5:

            self.zoom_out()

    def update_zoom_label(self):

        if self.zoom_mode == "fit":

            self.zoom_label.config(
                text="Fit"
            )

        else:

            self.zoom_label.config(
                text=f"{self.zoom_level * 100:.0f}%"
            )

    # ============================================================
    # DISPLAY IMAGE
    # ============================================================

    def display_image_on_preview(self):

        if self.crop_mode:
            return

        if self.current_image is None:
            return

        if self.preview_card is None:
            return

        card_width = self.preview_card.winfo_width()
        card_height = self.preview_card.winfo_height()

        if card_width <= 10 or card_height <= 10:
            return

        available_width = max(
            card_width - 45,
            300
        )

        available_height = max(
            card_height - 70,
            250
        )

        image_copy = self.current_image.copy()

        preview_resampling = (
            Image.Resampling.BILINEAR
            if self.is_slider_dragging
            else Image.Resampling.LANCZOS
        )

        if self.zoom_mode == "fit":

            image_copy.thumbnail(
                (
                    available_width,
                    available_height
                ),
                preview_resampling
            )

        else:

            new_width = max(
                1,
                int(
                    image_copy.width
                    * self.zoom_level
                )
            )

            new_height = max(
                1,
                int(
                    image_copy.height
                    * self.zoom_level
                )
            )

            image_copy = image_copy.resize(
                (
                    new_width,
                    new_height
                ),
                preview_resampling
            )

            # Prevent a huge image from consuming
            # excessive memory.

            max_dimension = (
                1200
                if self.is_slider_dragging
                else 2500
            )

            if (
                image_copy.width > max_dimension
                or image_copy.height > max_dimension
            ):

                image_copy.thumbnail(
                    (
                        max_dimension,
                        max_dimension
                    ),
                    preview_resampling
                )

        self.display_image = ImageTk.PhotoImage(
            image_copy
        )

        self.preview_label.config(
            image=self.display_image,
            text=""
        )

        self.preview_label.image = (
            self.display_image
        )

        self.update_zoom_label()

    # ============================================================
    # OPEN IMAGE
    # ============================================================

    def open_image(self):

        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                (
                    "Image Files",
                    "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif"
                ),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("BMP Files", "*.bmp"),
                ("GIF Files", "*.gif"),
                ("TIFF Files", "*.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:

            self.status.config(
                text="Opening image..."
            )

            self.root.update_idletasks()

            image = Image.open(
                file_path
            )

            if image.mode not in (
                "RGB",
                "RGBA",
                "L"
            ):

                image = image.convert(
                    "RGB"
                )

            self.original_image = image.copy()

            self.base_image = image.copy()

            self._preview_base_image = None
            self._preview_source_id = None
            self._cancel_live_preview()
            self.is_slider_dragging = False

            self.current_image = image.copy()

            self.image_path = file_path

            self.brightness_value = 0
            self.contrast_value = 0
            self.current_filter = "Original"

            self.saturation_value = 0
            self.hue_value = 0
            self.red_value = 0
            self.green_value = 0
            self.blue_value = 0
            self.gamma_value = 1.0
            self.threshold_value = 128
            self.advanced_mode = "None"

            self.zoom_mode = "fit"
            self.zoom_level = 1.0

            self.undo_stack.clear()
            self.redo_stack.clear()
            self.history.clear()
            self.processing_layers.clear()

            self.pending_slider_state = None
            self.pending_slider_type = None
            self.pending_processing_layer_description = None

            self.display_image_on_preview()

            self.status.config(
                text=(
                    f"Loaded: "
                    f"{file_path}   |   "
                    f"{image.width} × "
                    f"{image.height} px   |   "
                    f"{image.mode}"
                )
            )

            self.close_analysis_window()
            self.close_comparison_window()
            self.close_history_window()

        except Exception as error:

            messagebox.showerror(
                "Open Image Error",
                f"Unable to open image.\n\n{error}"
            )

            self.status.config(
                text="Unable to open image"
            )

    # ============================================================
    # SAVE IMAGE
    # ============================================================

    def save_image(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        self.commit_slider_history()

        format_map = {
            "PNG": (".png", [
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("BMP Files", "*.bmp"),
                ("TIFF Files", "*.tiff")
            ]),
            "JPEG": (".jpg", [
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("BMP Files", "*.bmp"),
                ("TIFF Files", "*.tiff")
            ]),
            "BMP": (".bmp", [
                ("BMP Files", "*.bmp"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("TIFF Files", "*.tiff")
            ]),
            "TIFF": (".tiff", [
                ("TIFF Files", "*.tiff"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("BMP Files", "*.bmp")
            ])
        }
        default_ext, file_types = format_map.get(
            self.settings_default_format, format_map["PNG"]
        )

        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=default_ext,
            filetypes=file_types
        )

        if not file_path:
            return

        try:

            self.status.config(
                text="Saving image..."
            )

            self.root.update_idletasks()

            # Always render the final image at full resolution before
            # saving. The live preview may be a reduced-resolution copy.
            self._cancel_live_preview()
            self.is_slider_dragging = False
            self.apply_adjustments(
                preview=False,
                refresh_secondary=False
            )

            image_to_save = self.current_image

            if file_path.lower().endswith(
                (".jpg", ".jpeg")
            ):

                if image_to_save.mode in (
                    "RGBA",
                    "LA",
                    "P"
                ):

                    image_to_save = image_to_save.convert(
                        "RGB"
                    )

            save_kwargs = {}
            if file_path.lower().endswith((".jpg", ".jpeg")):
                save_kwargs["quality"] = self.settings_jpeg_quality
                save_kwargs["optimize"] = True

            image_to_save.save(
                file_path,
                **save_kwargs
            )

            messagebox.showinfo(
                "Image Saved",
                f"Image successfully saved to:\n\n{file_path}"
            )

            self.status.config(
                text=f"Saved image: {file_path}"
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                f"Unable to save image.\n\n{error}"
            )

    # ============================================================
    # HISTORY STATE
    # ============================================================

    def get_editor_state(self):

        if self.base_image is None:
            return None

        return {
            "base_image": self.base_image.copy(),
            "brightness": self.brightness_value,
            "contrast": self.contrast_value,
            "filter": self.current_filter,
            "filter_strength": self.filter_strength,
            "filter_strengths": dict(self.filter_strengths),
            "saturation": self.saturation_value,
            "hue": self.hue_value,
            "red": self.red_value,
            "green": self.green_value,
            "blue": self.blue_value,
            "gamma": self.gamma_value,
            "threshold": self.threshold_value,
            "advanced_mode": self.advanced_mode
        }

    def restore_editor_state(
        self,
        state
    ):

        if state is None:
            return

        self.is_restoring_history = True

        self.base_image = (
            state["base_image"].copy()
        )

        self.brightness_value = (
            state["brightness"]
        )

        self.contrast_value = (
            state["contrast"]
        )

        self.current_filter = (
            state["filter"]
        )

        self.filter_strengths = dict(state.get("filter_strengths", {
            "Blur": 3.0,
            "Sharpen": 2.0,
            "Smooth": 3.0
        }))
        self.filter_strength = state.get(
            "filter_strength",
            self.filter_strengths.get(self.current_filter, 3.0)
        )

        self.saturation_value = state.get("saturation", 0)
        self.hue_value = state.get("hue", 0)
        self.red_value = state.get("red", 0)
        self.green_value = state.get("green", 0)
        self.blue_value = state.get("blue", 0)
        self.gamma_value = state.get("gamma", 1.0)
        self.threshold_value = state.get("threshold", 128)
        self.advanced_mode = state.get("advanced_mode", "None")

        if self.brightness_slider is not None:

            try:
                self.brightness_slider.set(
                    self.brightness_value
                )
            except tk.TclError:
                pass

        if self.contrast_slider is not None:

            try:
                self.contrast_slider.set(
                    self.contrast_value
                )
            except tk.TclError:
                pass

        self.apply_adjustments()

        self.is_restoring_history = False

    def record_history(
        self,
        description
    ):

        if self.base_image is None:
            return

        if self.is_restoring_history:
            return

        state = self.get_editor_state()

        # The operation mutates the editor immediately after this call.
        # Keep its description so apply_adjustments() can capture the
        # resulting state as a non-destructive stack checkpoint.
        self.pending_processing_layer_description = description

        self.undo_stack.append(
            state
        )

        self.redo_stack.clear()

        self.history.append(
            description
        )

        while len(self.undo_stack) > self.settings_max_history:
            self.undo_stack.pop(0)

        while len(self.history) > self.settings_max_history:
            self.history.pop(0)

        self.update_history_window()

    # ============================================================
    # UNDO
    # ============================================================

    def undo(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        self.commit_slider_history()

        if not self.undo_stack:

            self.status.config(
                text="Nothing to undo"
            )

            return

        current_state = self.get_editor_state()

        self.redo_stack.append(
            current_state
        )

        previous_state = self.undo_stack.pop()

        self.restore_editor_state(
            previous_state
        )

        if self.history:
            self.history.pop()

        self.update_history_window()

        self.status.config(
            text="Undo completed"
        )

    # ============================================================
    # REDO
    # ============================================================

    def redo(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        self.commit_slider_history()

        if not self.redo_stack:

            self.status.config(
                text="Nothing to redo"
            )

            return

        current_state = self.get_editor_state()

        self.undo_stack.append(
            current_state
        )

        next_state = self.redo_stack.pop()

        self.restore_editor_state(
            next_state
        )

        self.history.append(
            "Redo operation"
        )

        while len(self.history) > self.settings_max_history:
            self.history.pop(0)

        self.update_history_window()

        self.status.config(
            text="Redo completed"
        )

    # ============================================================
    # SLIDER HISTORY
    # ============================================================

    def begin_slider_history(
        self,
        slider_type
    ):

        if self.is_restoring_history:
            return

        if self.pending_slider_state is None:

            self.pending_slider_state = (
                self.get_editor_state()
            )

            self.pending_slider_type = (
                slider_type
            )

    def commit_slider_history(self):

        if self.pending_slider_state is None:
            return

        if self.is_restoring_history:

            self.pending_slider_state = None
            self.pending_slider_type = None

            return

        before_state = (
            self.pending_slider_state
        )

        current_state = (
            self.get_editor_state()
        )

        if (
            before_state is not None
            and current_state is not None
        ):

            self.undo_stack.append(
                before_state
            )

            self.redo_stack.clear()

            slider_descriptions = {
                "brightness": (
                    f"Brightness changed to "
                    f"{self.brightness_value:+d}"
                ),
                "contrast": (
                    f"Contrast changed to "
                    f"{self.contrast_value:+d}"
                ),
                "saturation": (
                    f"Saturation changed to "
                    f"{self.saturation_value:+d}"
                ),
                "hue": (
                    f"Hue changed to "
                    f"{self.hue_value:+d}"
                ),
                "red": (
                    f"Red channel changed to "
                    f"{self.red_value:+d}"
                ),
                "green": (
                    f"Green channel changed to "
                    f"{self.green_value:+d}"
                ),
                "blue": (
                    f"Blue channel changed to "
                    f"{self.blue_value:+d}"
                ),
                "gamma": (
                    f"Gamma changed to "
                    f"{self.gamma_value:.1f}"
                ),
                "threshold": (
                    f"Threshold changed to "
                    f"{self.threshold_value}"
                ),
                "filter_strength": (
                    f"{self.current_filter} strength changed to "
                    f"{self.filter_strength:.1f}"
                )
            }

            description = slider_descriptions.get(
                self.pending_slider_type,
                "Advanced adjustment changed"
            )

            self.history.append(
                description
            )

            while len(self.undo_stack) > self.settings_max_history:
                self.undo_stack.pop(0)

            while len(self.history) > self.settings_max_history:
                self.history.pop(0)

            self._add_processing_layer(
                description,
                current_state
            )

        self.pending_slider_state = None
        self.pending_slider_type = None

        self.update_history_window()

    # ============================================================
    # RESET IMAGE
    # ============================================================

    def reset_image(self):

        if self.original_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        self.commit_slider_history()

        if self.settings_confirm_reset:
            if not messagebox.askyesno(
                "Reset Image",
                "Reset the image and remove all current edits?"
            ):
                return

        self.record_history(
            "Reset image"
        )

        self.base_image = (
            self.original_image.copy()
        )

        self._preview_base_image = None
        self._preview_source_id = None
        self._cancel_live_preview()
        self.is_slider_dragging = False

        self.brightness_value = 0
        self.contrast_value = 0
        self.current_filter = "Original"

        # ========================================================
        # ADVANCED PROCESSING STATE
        # ========================================================

        self.saturation_value = 0
        self.hue_value = 0
        self.red_value = 0
        self.green_value = 0
        self.blue_value = 0
        self.gamma_value = 1.0
        self.threshold_value = 128
        self.advanced_mode = "None"

        if self.brightness_slider is not None:

            try:
                self.brightness_slider.set(0)
            except tk.TclError:
                pass

        if self.contrast_slider is not None:

            try:
                self.contrast_slider.set(0)
            except tk.TclError:
                pass

        self.zoom_fit()

        self.apply_adjustments()

        self.status.config(
            text="Image reset to original"
        )

    # ============================================================
    # HIDE PANELS
    # ============================================================

    def hide_adjustment_panels(self):

        self.brightness_frame.pack_forget()
        self.contrast_frame.pack_forget()
        self.filters_frame.pack_forget()
        self.transform_frame.pack_forget()
        self.advanced_frame.pack_forget()

        # Restore the normal panel height whenever switching tools.
        self.adjustment_container.configure(
            height=125
        )

    def set_active_tool(self, tool_id):
        """Highlight the currently selected main editing tool."""
        self.active_tool = tool_id

        for name, button in self.tool_buttons.items():
            try:
                button.configure(
                    style=("ToolActive.TButton" if name == tool_id else "Tool.TButton")
                )
            except tk.TclError:
                pass

    # ============================================================
    # BRIGHTNESS
    # ============================================================

    def show_brightness_control(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return
        self.set_active_tool("brightness")


        self.hide_adjustment_panels()

        self.brightness_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        for widget in (
            self.brightness_frame.winfo_children()
        ):
            widget.destroy()

        tk.Label(
            self.brightness_frame,
            text="Brightness",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        value_label = tk.Label(
            self.brightness_frame,
            text=f"{self.brightness_value:+d}",
            font=("Segoe UI", 10),
            bg=self.card_color,
            fg=self.secondary_text
        )

        value_label.grid(
            row=0,
            column=2,
            padx=10
        )

        ttk.Button(
            self.brightness_frame,
            text="Reset Brightness",
            command=self.reset_brightness
        ).grid(
            row=0,
            column=1,
            padx=10
        )

        self.brightness_slider = ttk.Scale(
            self.brightness_frame,
            from_=-100,
            to=100,
            orient="horizontal",
            command=lambda value: (
                self.begin_slider_history(
                    "brightness"
                ),
                self.apply_brightness(
                    value
                ),
                value_label.config(
                    text=f"{int(float(value)):+d}"
                )
            )
        )

        self.brightness_slider.set(
            self.brightness_value
        )

        self.brightness_slider.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=8
        )

        self.brightness_slider.bind(
            "<ButtonRelease-1>",
            lambda event:
            self.finish_slider_edit()
        )

        self.brightness_frame.grid_columnconfigure(
            0,
            weight=1
        )

    def apply_brightness(
        self,
        value
    ):

        if self.base_image is None:
            return

        self.brightness_value = int(
            float(value)
        )

        self._schedule_live_preview()

        self.status.config(
            text=(
                f"Brightness: "
                f"{self.brightness_value:+d}"
            )
        )

    def reset_brightness(self):

        self._cancel_live_preview()
        self.is_slider_dragging = False
        self.commit_slider_history()

        self.record_history(
            "Reset brightness"
        )

        self.brightness_value = 0

        if self.brightness_slider is not None:
            self.brightness_slider.set(0)

        self.apply_adjustments()

    # ============================================================
    # CONTRAST
    # ============================================================

    def show_contrast_control(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return
        self.set_active_tool("contrast")


        self.hide_adjustment_panels()

        self.contrast_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        for widget in (
            self.contrast_frame.winfo_children()
        ):
            widget.destroy()

        tk.Label(
            self.contrast_frame,
            text="Contrast",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        value_label = tk.Label(
            self.contrast_frame,
            text=f"{self.contrast_value:+d}",
            font=("Segoe UI", 10),
            bg=self.card_color,
            fg=self.secondary_text
        )

        value_label.grid(
            row=0,
            column=2,
            padx=10
        )

        ttk.Button(
            self.contrast_frame,
            text="Reset Contrast",
            command=self.reset_contrast
        ).grid(
            row=0,
            column=1,
            padx=10
        )

        self.contrast_slider = ttk.Scale(
            self.contrast_frame,
            from_=-100,
            to=100,
            orient="horizontal",
            command=lambda value: (
                self.begin_slider_history(
                    "contrast"
                ),
                self.apply_contrast(
                    value
                ),
                value_label.config(
                    text=f"{int(float(value)):+d}"
                )
            )
        )

        self.contrast_slider.set(
            self.contrast_value
        )

        self.contrast_slider.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=8
        )

        self.contrast_slider.bind(
            "<ButtonRelease-1>",
            lambda event:
            self.finish_slider_edit()
        )

        self.contrast_frame.grid_columnconfigure(
            0,
            weight=1
        )

    def apply_contrast(
        self,
        value
    ):

        if self.base_image is None:
            return

        self.contrast_value = int(
            float(value)
        )

        self._schedule_live_preview()

        self.status.config(
            text=(
                f"Contrast: "
                f"{self.contrast_value:+d}"
            )
        )

    def reset_contrast(self):

        self._cancel_live_preview()
        self.is_slider_dragging = False
        self.commit_slider_history()

        self.record_history(
            "Reset contrast"
        )

        self.contrast_value = 0

        if self.contrast_slider is not None:
            self.contrast_slider.set(0)

        self.apply_adjustments()

    # ============================================================
    # FILTERS
    # ============================================================

    def show_filters_control(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return
        self.set_active_tool("filters")


        self.hide_adjustment_panels()

        self.filters_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=8
        )

        for widget in self.filters_frame.winfo_children():
            widget.destroy()

        filters = [
            "Original",
            "Grayscale",
            "Sepia",
            "Negative",
            "Blur",
            "Sharpen",
            "Edge Enhance",
            "Emboss",
            "Smooth",
            "Contour"
        ]

        # These filters have adjustable strength.  Their sliders stay
        # hidden until the corresponding filter button is clicked.
        adjustable = {
            "Blur": (0.0, 20.0, 3.0),
            "Sharpen": (0.0, 5.0, 2.0),
            "Smooth": (0.0, 10.0, 3.0)
        }

        self.filter_strength_sliders = {}
        self.filter_strength_value_labels = {}
        self.filter_strength_cells = {}

        for index, filter_name in enumerate(filters):

            row = index // 5
            column = index % 5

            cell = tk.Frame(
                self.filters_frame,
                bg=self.card_color,
                highlightthickness=0
            )

            cell.grid(
                row=row,
                column=column,
                padx=4,
                pady=4,
                sticky="ew"
            )

            # Keep the filter button itself the same normal size.
            button = ttk.Button(
                cell,
                text=filter_name,
                command=lambda name=filter_name:
                self.apply_filter(name)
            )

            button.pack(
                fill="x"
            )

            if filter_name in adjustable:

                minimum, maximum, default = adjustable[filter_name]

                if filter_name not in self.filter_strengths:
                    self.filter_strengths[filter_name] = default

                value = max(
                    minimum,
                    min(maximum, float(self.filter_strengths[filter_name]))
                )
                self.filter_strengths[filter_name] = value

                value_label = tk.Label(
                    cell,
                    text=f"{value:.1f}",
                    font=("Segoe UI", 8),
                    bg=self.card_color,
                    fg=self.secondary_text
                )

                slider = ttk.Scale(
                    cell,
                    from_=minimum,
                    to=maximum,
                    orient="horizontal",
                    command=lambda raw_value, name=filter_name, label=value_label:
                    self.update_embedded_filter_strength(
                        name,
                        raw_value,
                        label
                    )
                )

                slider.set(value)

                slider.pack(
                    fill="x",
                    padx=6,
                    pady=(2, 4)
                )

                slider.bind(
                    "<ButtonPress-1>",
                    lambda event, name=filter_name:
                    self.begin_embedded_filter_slider(name)
                )

                slider.bind(
                    "<ButtonRelease-1>",
                    lambda event:
                    self.finish_slider_edit()
                )

                # Hide the adjustment controls initially.  They are
                # revealed only after this specific filter is selected.
                value_label.pack_forget()
                slider.pack_forget()

                self.filter_strength_sliders[filter_name] = slider
                self.filter_strength_value_labels[filter_name] = value_label
                self.filter_strength_cells[filter_name] = cell

        for column in range(5):
            self.filters_frame.grid_columnconfigure(
                column,
                weight=1
            )

        # Do not stretch the filter buttons vertically when a slider
        # appears.  This keeps the original button size/design.
        self.filters_frame.grid_rowconfigure(0, weight=0)
        self.filters_frame.grid_rowconfigure(1, weight=0)

        # Show a strength slider only for the currently selected
        # adjustable filter.
        if self.current_filter in self.filter_strength_sliders:

            slider = self.filter_strength_sliders[self.current_filter]
            label = self.filter_strength_value_labels[self.current_filter]

            label.pack(
                pady=(2, 0)
            )

            slider.pack(
                fill="x",
                padx=6,
                pady=(0, 4)
            )

            self.filter_strength_slider = slider

        else:
            self.filter_strength_slider = None

        # Keep the existing compact filter-panel height.  The selected
        # filter's slider is allowed to add only the space it needs.
        self.adjustment_container.configure(
            height=150
        )

    def begin_embedded_filter_slider(self, filter_name):

        if self.current_filter != filter_name:
            self.commit_slider_history()
            self.record_history(
                f"Applied filter: {filter_name}"
            )
            self.current_filter = filter_name
            self.filter_strength = self.filter_strengths.get(
                filter_name,
                3.0
            )
            self.apply_adjustments()

        self.begin_slider_history("filter_strength")
        self.is_slider_dragging = True

    def update_embedded_filter_strength(
        self,
        filter_name,
        value,
        value_label
    ):

        if self.base_image is None:
            return

        value = float(value)

        self.filter_strengths[filter_name] = value

        if self.current_filter == filter_name:
            self.filter_strength = value
            self._schedule_live_preview()

            self.status.config(
                text=(
                    f"{filter_name}: "
                    f"{value:.1f}"
                )
            )

        value_label.config(
            text=f"{value:.1f}"
        )

        self.filter_strength_slider = self.filter_strength_sliders.get(
            self.current_filter
        )

    def apply_filter_strength(self, value):

        if self.base_image is None:
            return

        self.filter_strength = float(value)

        self._schedule_live_preview()

        self.status.config(
            text=(
                f"{self.current_filter}: "
                f"{self.filter_strength:.1f}"
            )
        )

    def reset_filter_strength(self):

        self._cancel_live_preview()
        self.is_slider_dragging = False
        self.commit_slider_history()

        if self.current_filter == "Blur":
            value = 3.0
        elif self.current_filter == "Sharpen":
            value = 2.0
        elif self.current_filter == "Smooth":
            value = 3.0
        else:
            return

        self.record_history(
            f"Reset {self.current_filter} strength"
        )

        self.filter_strength = value
        self.filter_strengths[self.current_filter] = value

        slider = self.filter_strength_sliders.get(
            self.current_filter
        ) if hasattr(self, "filter_strength_sliders") else None

        if slider is not None:
            try:
                slider.set(value)
            except tk.TclError:
                pass

        label = self.filter_strength_value_labels.get(
            self.current_filter
        ) if hasattr(self, "filter_strength_value_labels") else None

        if label is not None:
            label.config(text=f"{value:.1f}")

        self.apply_adjustments()

    def apply_filter(
        self,
        filter_name
    ):

        if self.base_image is None:
            return

        self.commit_slider_history()

        if filter_name == self.current_filter:
            self.show_filters_control()
            return

        self.record_history(
            f"Applied filter: {filter_name}"
        )

        self.current_filter = filter_name

        if filter_name in self.filter_strengths:
            self.filter_strength = self.filter_strengths[filter_name]

        self.apply_adjustments()

        self.status.config(
            text=f"Filter: {filter_name}"
        )

        self.show_filters_control()

    # ============================================================
    # APPLY ADJUSTMENTS
    # ============================================================

    def _cancel_live_preview(self):

        if self.live_preview_after_id is not None:

            try:
                self.root.after_cancel(
                    self.live_preview_after_id
                )
            except (tk.TclError, ValueError):
                pass

            self.live_preview_after_id = None

    def _get_preview_base_image(self):

        if self.base_image is None:
            return None

        source_id = id(self.base_image)

        if (
            self._preview_base_image is None
            or self._preview_source_id != source_id
        ):

            preview = self.base_image.copy()

            if (
                preview.width > self.preview_max_dimension
                or preview.height > self.preview_max_dimension
            ):

                preview.thumbnail(
                    (
                        self.preview_max_dimension,
                        self.preview_max_dimension
                    ),
                    Image.Resampling.BILINEAR
                )

            self._preview_base_image = preview
            self._preview_source_id = source_id

        # The cached preview is never modified in-place by the processing
        # pipeline, so returning it directly avoids an unnecessary full
        # image copy on every slider event.
        return self._preview_base_image

    def _schedule_live_preview(self):

        if self.base_image is None:
            return

        self.is_slider_dragging = True

        if not self.settings_live_preview:
            self._cancel_live_preview()
            return
        self._cancel_live_preview()

        # Coalesce rapid slider events. Tkinter can generate many
        # callbacks while the mouse is moving; processing only the
        # newest value prevents a backlog and keeps the UI responsive.
        self.live_preview_after_id = self.root.after(50, self._run_live_preview)

    def _run_live_preview(self):

        self.live_preview_after_id = None

        if self.base_image is None:
            return

        self.apply_adjustments(
            preview=True,
            refresh_secondary=False
        )

    def finish_slider_edit(self):

        self._cancel_live_preview()
        self.is_slider_dragging = False

        if self.base_image is None:
            self.commit_slider_history()
            return

        # One full-resolution render when the user releases the slider.
        self.apply_adjustments(
            preview=False,
            refresh_secondary=True
        )

        self.commit_slider_history()

    def apply_adjustments(
        self,
        preview=False,
        refresh_secondary=True
    ):

        if self.base_image is None:
            return

        if preview:
            image = self._get_preview_base_image()
        else:
            image = self.base_image.copy()

        # --------------------------------------------------------
        # NUMPY BRIGHTNESS / CONTRAST
        # --------------------------------------------------------

        if image.mode in (
            "RGB",
            "RGBA",
            "L"
        ):

            image_array = np.array(
                image
            ).astype(
                np.int16
            )

            if image.mode == "RGBA":

                rgb = image_array[:, :, :3]

                alpha = image_array[:, :, 3]

                rgb = (
                    rgb
                    + self.brightness_value
                )

                rgb = np.clip(
                    rgb,
                    0,
                    255
                )

                factor = (
                    100
                    + self.contrast_value
                ) / 100

                rgb = (
                    (rgb - 128)
                    * factor
                ) + 128

                rgb = np.clip(
                    rgb,
                    0,
                    255
                )

                image_array = np.dstack(
                    (
                        rgb,
                        alpha
                    )
                ).astype(
                    np.uint8
                )

            else:

                image_array = (
                    image_array
                    + self.brightness_value
                )

                image_array = np.clip(
                    image_array,
                    0,
                    255
                )

                factor = (
                    100
                    + self.contrast_value
                ) / 100

                image_array = (
                    (image_array - 128)
                    * factor
                ) + 128

                image_array = np.clip(
                    image_array,
                    0,
                    255
                )

                image_array = image_array.astype(
                    np.uint8
                )

            image = Image.fromarray(
                image_array
            )

        # --------------------------------------------------------
        # ADVANCED RGB / HSV / GAMMA PROCESSING
        # --------------------------------------------------------

        if image.mode in ("RGB", "RGBA"):

            has_alpha = image.mode == "RGBA"

            if has_alpha:
                alpha_channel = image.getchannel("A")
                rgb_image = image.convert("RGB")
            else:
                alpha_channel = None
                rgb_image = image.convert("RGB")

            rgb_array = np.array(
                rgb_image
            ).astype(
                np.int16
            )

            # RGB channel adjustment
            channel_adjustments = np.array(
                [
                    self.red_value,
                    self.green_value,
                    self.blue_value
                ],
                dtype=np.int16
            )

            rgb_array = rgb_array + channel_adjustments
            rgb_array = np.clip(
                rgb_array,
                0,
                255
            ).astype(
                np.uint8
            )

            rgb_image = Image.fromarray(
                rgb_array,
                "RGB"
            )

            # Saturation and hue adjustment
            if (
                self.saturation_value != 0
                or self.hue_value != 0
            ):

                hsv_image = rgb_image.convert("HSV")
                hsv_array = np.array(
                    hsv_image
                ).astype(
                    np.int16
                )

                if self.saturation_value != 0:

                    saturation_factor = (
                        100
                        + self.saturation_value
                    ) / 100.0

                    hsv_array[:, :, 1] = np.clip(
                        hsv_array[:, :, 1]
                        * saturation_factor,
                        0,
                        255
                    )

                if self.hue_value != 0:

                    hue_shift = int(
                        self.hue_value
                        * 255
                        / 360
                    )

                    hsv_array[:, :, 0] = (
                        hsv_array[:, :, 0]
                        + hue_shift
                    ) % 256

                hsv_array = hsv_array.astype(
                    np.uint8
                )

                rgb_image = Image.fromarray(
                    hsv_array,
                    "HSV"
                ).convert("RGB")

            # Gamma correction
            if abs(self.gamma_value - 1.0) > 0.001:

                gamma_array = np.array(
                    rgb_image
                ).astype(
                    np.float32
                ) / 255.0

                gamma_array = np.power(
                    gamma_array,
                    1.0 / self.gamma_value
                )

                gamma_array = (
                    gamma_array
                    * 255.0
                )

                gamma_array = np.clip(
                    gamma_array,
                    0,
                    255
                ).astype(
                    np.uint8
                )

                rgb_image = Image.fromarray(
                    gamma_array,
                    "RGB"
                )

            if has_alpha:

                image = Image.merge(
                    "RGBA",
                    (
                        rgb_image.getchannel("R"),
                        rgb_image.getchannel("G"),
                        rgb_image.getchannel("B"),
                        alpha_channel
                    )
                )

            else:

                image = rgb_image

        # --------------------------------------------------------
        # ADVANCED PROCESSING MODES
        # --------------------------------------------------------

        if self.advanced_mode == "Histogram Equalization":

            if image.mode == "RGBA":

                rgb = image.convert("RGB")
                alpha = image.getchannel("A")

                equalized = ImageOps.equalize(
                    rgb
                )

                image = Image.merge(
                    "RGBA",
                    (
                        equalized.getchannel("R"),
                        equalized.getchannel("G"),
                        equalized.getchannel("B"),
                        alpha
                    )
                )

            else:

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image = ImageOps.equalize(
                    image
                )

        elif self.advanced_mode == "Auto Enhance":

            if image.mode == "RGBA":

                rgb = image.convert("RGB")
                alpha = image.getchannel("A")

                enhanced = ImageOps.autocontrast(
                    rgb
                )

                image = Image.merge(
                    "RGBA",
                    (
                        enhanced.getchannel("R"),
                        enhanced.getchannel("G"),
                        enhanced.getchannel("B"),
                        alpha
                    )
                )

            else:

                image = ImageOps.autocontrast(
                    image
                )

        elif self.advanced_mode == "Threshold":

            grayscale = ImageOps.grayscale(
                image
            )

            threshold_array = np.array(
                grayscale
            )

            threshold_array = np.where(
                threshold_array >= self.threshold_value,
                255,
                0
            ).astype(
                np.uint8
            )

            image = Image.fromarray(
                threshold_array,
                "L"
            )

        # --------------------------------------------------------
        # PILLOW FILTERS
        # --------------------------------------------------------

        if self.current_filter == "Grayscale":

            image = ImageOps.grayscale(
                image
            ).convert("RGB")

        elif self.current_filter == "Sepia":

            grayscale = ImageOps.grayscale(
                image
            )

            image = ImageOps.colorize(
                grayscale,
                black="#3b2415",
                white="#f5d7a1"
            )

        elif self.current_filter == "Negative":

            if image.mode == "RGBA":

                rgb = image.convert(
                    "RGB"
                )

                alpha = image.getchannel(
                    "A"
                )

                inverted = ImageOps.invert(
                    rgb
                )

                image = Image.merge(
                    "RGBA",
                    (
                        inverted.getchannel("R"),
                        inverted.getchannel("G"),
                        inverted.getchannel("B"),
                        alpha
                    )
                )

            else:

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image = ImageOps.invert(
                    image
                )

        elif self.current_filter == "Blur":

            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=max(0.0, self.filter_strength)
                )
            )

        elif self.current_filter == "Sharpen":

            if self.filter_strength > 0:
                image = image.filter(
                    ImageFilter.UnsharpMask(
                        radius=2,
                        percent=int(self.filter_strength * 100),
                        threshold=3
                    )
                )

        elif self.current_filter == "Edge Enhance":

            image = image.filter(
                ImageFilter.EDGE_ENHANCE
            )

        elif self.current_filter == "Emboss":

            image = image.filter(
                ImageFilter.EMBOSS
            )

        elif self.current_filter == "Smooth":

            image = image.filter(
                ImageFilter.BoxBlur(
                    radius=max(0.0, self.filter_strength)
                )
            )

        elif self.current_filter == "Contour":

            image = image.filter(
                ImageFilter.CONTOUR
            )

        self.current_image = image

        self.display_image_on_preview()

        if refresh_secondary:
            self.refresh_secondary_windows()

        if not preview:
            self._finalize_pending_processing_layer()

    # ============================================================
    # ADVANCED PROCESSING
    # ============================================================

    def show_advanced_control(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return
        self.set_active_tool("advanced")


        self.hide_adjustment_panels()

        self.advanced_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=6
        )

        for widget in self.advanced_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self.advanced_frame,
            text="Advanced Processing",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).grid(
            row=0,
            column=0,
            padx=5,
            sticky="w"
        )

        slider_specs = [
            ("Saturation", "saturation", -100, 100, self.saturation_value, 0),
            ("Hue", "hue", -180, 180, self.hue_value, 1),
            ("Red", "red", -100, 100, self.red_value, 2),
            ("Green", "green", -100, 100, self.green_value, 3),
            ("Blue", "blue", -100, 100, self.blue_value, 4),
            ("Gamma", "gamma", 0.1, 3.0, self.gamma_value, 5),
            ("Threshold", "threshold", 0, 255, self.threshold_value, 6)
        ]

        self.advanced_sliders = {}

        for title, key, minimum, maximum, current, column in slider_specs:

            cell = tk.Frame(
                self.advanced_frame,
                bg=self.card_color
            )

            cell.grid(
                row=0,
                column=column,
                padx=3,
                sticky="nsew"
            )

            label_frame = tk.Frame(
                cell,
                bg=self.card_color
            )

            label_frame.pack(
                fill="x"
            )

            tk.Label(
                label_frame,
                text=title,
                font=("Segoe UI", 8, "bold"),
                bg=self.card_color,
                fg=self.text_color
            ).pack(
                side="left"
            )

            value_label = tk.Label(
                label_frame,
                text=(
                    f"{current:.1f}"
                    if key == "gamma"
                    else str(int(current))
                ),
                font=("Segoe UI", 8),
                bg=self.card_color,
                fg=self.secondary_text
            )

            value_label.pack(
                side="right"
            )

            slider = ttk.Scale(
                cell,
                from_=minimum,
                to=maximum,
                orient="horizontal",
                command=lambda value, k=key, label=value_label:
                    self.update_advanced_slider(
                        k,
                        value,
                        label
                    )
            )

            slider.set(current)

            slider.pack(
                fill="x",
                pady=(3, 0)
            )

            slider.bind(
                "<ButtonRelease-1>",
                lambda event:
                self.finish_slider_edit()
            )

            self.advanced_sliders[key] = slider

        for column in range(7):

            self.advanced_frame.grid_columnconfigure(
                column,
                weight=1
            )

        mode_frame = tk.Frame(
            self.advanced_frame,
            bg=self.card_color
        )

        mode_frame.grid(
            row=1,
            column=0,
            columnspan=7,
            sticky="ew",
            pady=(7, 0)
        )

        tk.Label(
            mode_frame,
            text="Processing Mode:",
            font=("Segoe UI", 8, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            side="left",
            padx=(2, 7)
        )

        modes = [
            "None",
            "Histogram Equalization",
            "Auto Enhance",
            "Threshold"
        ]

        for mode in modes:

            ttk.Button(
                mode_frame,
                text=mode,
                command=lambda name=mode:
                self.set_advanced_mode(name)
            ).pack(
                side="left",
                padx=2
            )

        ttk.Button(
            mode_frame,
            text="Reset Advanced",
            command=self.reset_advanced
        ).pack(
            side="right",
            padx=2
        )

        tk.Label(
            self.advanced_frame,
            text="Live preview uses a reduced-resolution copy for smoother editing. Full resolution is used when saving.",
            font=("Segoe UI", 7),
            bg=self.card_color,
            fg=self.secondary_text
        ).grid(
            row=2,
            column=0,
            columnspan=7,
            sticky="w",
            padx=5,
            pady=(5, 0)
        )

    def update_advanced_slider(
        self,
        key,
        value,
        value_label
    ):

        if self.base_image is None:
            return

        self.begin_slider_history(key)

        numeric_value = float(value)

        if key == "saturation":
            self.saturation_value = int(numeric_value)
            value_label.config(
                text=f"{self.saturation_value:+d}"
            )

        elif key == "hue":
            self.hue_value = int(numeric_value)
            value_label.config(
                text=f"{self.hue_value:+d}"
            )

        elif key == "red":
            self.red_value = int(numeric_value)
            value_label.config(
                text=f"{self.red_value:+d}"
            )

        elif key == "green":
            self.green_value = int(numeric_value)
            value_label.config(
                text=f"{self.green_value:+d}"
            )

        elif key == "blue":
            self.blue_value = int(numeric_value)
            value_label.config(
                text=f"{self.blue_value:+d}"
            )

        elif key == "gamma":
            self.gamma_value = round(
                max(0.1, min(3.0, numeric_value)),
                1
            )
            value_label.config(
                text=f"{self.gamma_value:.1f}"
            )

        elif key == "threshold":
            self.threshold_value = int(numeric_value)
            value_label.config(
                text=str(self.threshold_value)
            )

        self._schedule_live_preview()

        self.status.config(
            text=f"Advanced: {key.title()} adjusted"
        )

    def set_advanced_mode(
        self,
        mode
    ):

        if self.base_image is None:
            return

        self.commit_slider_history()

        if mode == self.advanced_mode:
            return

        self.record_history(
            f"Advanced mode: {mode}"
        )

        self.advanced_mode = mode

        self.apply_adjustments()

        self.status.config(
            text=f"Advanced mode: {mode}"
        )

    def reset_advanced(self):

        if self.base_image is None:
            return

        self._cancel_live_preview()
        self.is_slider_dragging = False
        self.commit_slider_history()

        self.record_history(
            "Reset advanced processing"
        )

        self.saturation_value = 0
        self.hue_value = 0
        self.red_value = 0
        self.green_value = 0
        self.blue_value = 0
        self.gamma_value = 1.0
        self.threshold_value = 128
        self.advanced_mode = "None"

        if self.advanced_frame is not None:

            for key, value in {
                "saturation": 0,
                "hue": 0,
                "red": 0,
                "green": 0,
                "blue": 0,
                "gamma": 1.0,
                "threshold": 128
            }.items():

                slider = self.advanced_sliders.get(key)

                if slider is not None:

                    try:
                        slider.set(value)
                    except tk.TclError:
                        pass

        self.apply_adjustments()

        self.status.config(
            text="Advanced processing reset"
        )

    # ============================================================
    # TRANSFORM PANEL
    # ============================================================

    def show_transform_control(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return
        self.set_active_tool("transform")


        self.hide_adjustment_panels()

        self.transform_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        for widget in (
            self.transform_frame.winfo_children()
        ):
            widget.destroy()

        buttons = [
            (
                "Flip Horizontal",
                self.flip_horizontal
            ),
            (
                "Flip Vertical",
                self.flip_vertical
            ),
            (
                "Rotate 90°",
                self.rotate_90
            ),
            (
                "Rotate 180°",
                self.rotate_180
            ),
            (
                "Rotate 270°",
                self.rotate_270
            ),
            (
                "Custom Rotation",
                self.show_custom_rotation
            ),
            (
                "Resize",
                self.show_resize
            ),
            (
                "Crop",
                self.show_crop
            ),
            (
                "Reset Transform",
                self.reset_transform
            )
        ]

        for index, (
            text,
            command
        ) in enumerate(buttons):

            row = index // 5
            column = index % 5

            ttk.Button(
                self.transform_frame,
                text=text,
                command=command
            ).grid(
                row=row,
                column=column,
                padx=4,
                pady=4,
                sticky="ew"
            )

        for column in range(5):

            self.transform_frame.grid_columnconfigure(
                column,
                weight=1
            )

    # ============================================================
    # TRANSFORMS
    # ============================================================

    def apply_transform_result(
        self,
        image,
        description
    ):

        self.commit_slider_history()

        self.record_history(
            description
        )

        self.base_image = image.copy()

        self._preview_base_image = None
        self._preview_source_id = None

        # Show the transformed result through the fast preview pipeline
        # immediately. Full-resolution rendering is still performed when
        # the image is saved, so editing stays responsive on large images.
        self.is_slider_dragging = True
        self.apply_adjustments(
            preview=True,
            refresh_secondary=False
        )
        self.is_slider_dragging = False

        self.status.config(
            text=(
                f"{description} | "
                f"Size: "
                f"{self.base_image.width} × "
                f"{self.base_image.height} px"
            )
        )

    def flip_horizontal(self):

        self.apply_transform_result(
            ImageOps.mirror(
                self.base_image
            ),
            "Flip horizontal"
        )

    def flip_vertical(self):

        self.apply_transform_result(
            ImageOps.flip(
                self.base_image
            ),
            "Flip vertical"
        )

    def rotate_90(self):

        self.apply_transform_result(
            self.base_image.rotate(
                90,
                expand=True
            ),
            "Rotate 90°"
        )

    def rotate_180(self):

        self.apply_transform_result(
            self.base_image.rotate(
                180,
                expand=True
            ),
            "Rotate 180°"
        )

    def rotate_270(self):

        self.apply_transform_result(
            self.base_image.rotate(
                270,
                expand=True
            ),
            "Rotate 270°"
        )

    def reset_transform(self):

        if self.original_image is None:
            return

        self.commit_slider_history()

        self.record_history(
            "Reset transformations"
        )

        self.base_image = (
            self.original_image.copy()
        )

        self.apply_adjustments()

        self.status.config(
            text="Geometric transformations reset"
        )

    # ============================================================
    # CUSTOM ROTATION
    # ============================================================

    def show_custom_rotation(self):

        if self.base_image is None:
            return

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Custom Rotation"
        )

        window.geometry(
            "800x650"
        )

        window.transient(
            self.root
        )

        window.grab_set()

        tk.Label(
            window,
            text="Custom Rotation",
            font=("Segoe UI", 14, "bold")
        ).pack(
            pady=10
        )

        canvas = tk.Canvas(
            window,
            bg="#202020",
            highlightthickness=0
        )

        canvas.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        value_label = tk.Label(
            window,
            text="0°",
            font=("Segoe UI", 10, "bold")
        )

        value_label.pack()

        preview_reference = {
            "image": None
        }

        def update_preview(value):

            angle = float(value)

            # Custom rotation is interactive, so rotate the reduced preview
            # rather than the full-resolution image on every slider tick.
            preview_source = self._get_preview_base_image()

            rotated = preview_source.rotate(
                -angle,
                expand=True,
                resample=Image.Resampling.BILINEAR
            )

            preview = rotated

            preview.thumbnail(
                (650, 450),
                Image.Resampling.BILINEAR
            )

            preview_reference["image"] = (
                ImageTk.PhotoImage(
                    preview
                )
            )

            canvas.delete("all")

            canvas.create_image(
                canvas.winfo_width() // 2,
                canvas.winfo_height() // 2,
                image=preview_reference["image"],
                anchor="center"
            )

            value_label.config(
                text=f"{angle:+.0f}°"
            )

        slider = ttk.Scale(
            window,
            from_=-180,
            to=180,
            orient="horizontal",
            command=update_preview
        )

        slider.set(0)

        slider.pack(
            fill="x",
            padx=30,
            pady=10
        )

        buttons = tk.Frame(
            window
        )

        buttons.pack(
            pady=10
        )

        def apply_rotation():

            angle = float(
                slider.get()
            )

            rotated = self.base_image.rotate(
                -angle,
                expand=True,
                resample=Image.Resampling.BICUBIC
            )

            self.apply_transform_result(
                rotated,
                f"Custom rotation {angle:+.0f}°"
            )

            window.destroy()

        ttk.Button(
            buttons,
            text="Apply Rotation",
            command=apply_rotation
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Cancel",
            command=window.destroy
        ).pack(
            side="left",
            padx=5
        )

        window.after(
            100,
            lambda: update_preview(0)
        )

    # ============================================================
    # RESIZE
    # ============================================================

    def show_resize(self):

        if self.base_image is None:
            return

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Resize Image"
        )

        window.geometry(
            "850x700"
        )

        window.transient(
            self.root
        )

        window.grab_set()

        tk.Label(
            window,
            text="Resize Image",
            font=("Segoe UI", 14, "bold")
        ).pack(
            pady=10
        )

        canvas = tk.Canvas(
            window,
            bg="#202020",
            highlightthickness=0
        )

        canvas.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        info_label = tk.Label(
            window,
            text="",
            font=("Segoe UI", 10)
        )

        info_label.pack(
            pady=5
        )

        original_width = self.base_image.width
        original_height = self.base_image.height

        scale = 1.0

        preview_reference = {
            "image": None
        }

        def redraw():

            new_width = max(
                1,
                int(original_width * scale)
            )

            new_height = max(
                1,
                int(original_height * scale)
            )

            # Interactive resize preview uses a reduced source image.
            # The full-resolution resize is performed only when Apply is clicked.
            preview_source = self._get_preview_base_image()

            preview_width = max(
                1,
                int(preview_source.width * scale)
            )
            preview_height = max(
                1,
                int(preview_source.height * scale)
            )

            preview = preview_source.resize(
                (
                    preview_width,
                    preview_height
                ),
                Image.Resampling.BILINEAR
            )

            preview.thumbnail(
                (680, 480),
                Image.Resampling.BILINEAR
            )

            preview_reference["image"] = (
                ImageTk.PhotoImage(
                    preview
                )
            )

            canvas.delete("all")

            canvas.create_image(
                canvas.winfo_width() // 2,
                canvas.winfo_height() // 2,
                image=preview_reference["image"],
                anchor="center"
            )

            info_label.config(
                text=(
                    f"Size: "
                    f"{new_width} × "
                    f"{new_height} px   |   "
                    f"Scale: "
                    f"{scale * 100:.0f}%"
                )
            )

        def increase():

            nonlocal scale

            scale = min(
                5.0,
                scale + 0.1
            )

            redraw()

        def decrease():

            nonlocal scale

            scale = max(
                0.1,
                scale - 0.1
            )

            redraw()

        def apply_resize():

            new_width = max(
                1,
                int(original_width * scale)
            )

            new_height = max(
                1,
                int(original_height * scale)
            )

            resized = self.base_image.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.LANCZOS
            )

            self.apply_transform_result(
                resized,
                (
                    f"Resize to "
                    f"{new_width} × "
                    f"{new_height}"
                )
            )

            window.destroy()

        buttons = tk.Frame(
            window
        )

        buttons.pack(
            pady=10
        )

        ttk.Button(
            buttons,
            text="−",
            width=5,
            command=decrease
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="+",
            width=5,
            command=increase
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Apply Resize",
            command=apply_resize
        ).pack(
            side="left",
            padx=15
        )

        ttk.Button(
            buttons,
            text="Cancel",
            command=window.destroy
        ).pack(
            side="left",
            padx=5
        )

        window.after(
            100,
            redraw
        )

    # ============================================================
    # CROP - INLINE EDITOR
    # ============================================================

    def show_crop(self):

        if self.base_image is None:
            return

        if self.crop_mode:
            return

        self.commit_slider_history()

        self._cancel_live_preview()
        self.is_slider_dragging = False
        self.apply_adjustments(
            preview=False,
            refresh_secondary=False
        )

        self.hide_adjustment_panels()

        self.crop_mode = True
        self.crop_canvas = tk.Canvas(
            self.preview_card,
            bg="#202020",
            highlightthickness=0
        )

        self.preview_label.pack_forget()
        self.crop_canvas.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(5, 10)
        )

        # Phone/gallery-style crop state:
        # - the crop frame keeps the original image aspect ratio
        # - the mouse wheel zooms the image underneath the fixed frame
        # - dragging moves the image
        # - applying the crop resizes the selected area back to the
        #   original image dimensions, so the final size does not change
        self.crop_zoom = 1.0
        self.crop_pan_x = 0.0
        self.crop_pan_y = 0.0
        self.crop_frame = None
        self.crop_image_bounds = None
        self.crop_handles = {}
        self.crop_handle_size = 7
        self.crop_min_size = 80
        self.crop_drag_data = {}

        self.crop_canvas.bind(
            "<Configure>",
            lambda event: self._redraw_crop_editor()
        )

        # Crop box controls + image panning.
        # - Drag inside the crop box: move the crop box.
        # - Drag a corner handle: resize the crop box.
        # - Drag outside the crop box: move the image underneath it.
        self.crop_canvas.bind(
            "<ButtonPress-1>",
            self._start_crop_interaction
        )
        self.crop_canvas.bind(
            "<B1-Motion>",
            self._drag_crop_interaction
        )
        self.crop_canvas.bind(
            "<ButtonRelease-1>",
            self._stop_crop_interaction
        )

        self.crop_canvas.bind(
            "<MouseWheel>",
            self._crop_mouse_wheel
        )
        self.crop_canvas.bind(
            "<Button-4>",
            lambda event: self._change_crop_zoom(1)
        )
        self.crop_canvas.bind(
            "<Button-5>",
            lambda event: self._change_crop_zoom(-1)
        )

        self._redraw_crop_editor()

        self.transform_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        for widget in self.transform_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self.transform_frame,
            text="Crop Image",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            padx=5,
            pady=(0, 2)
        )

        crop_info_label = tk.Label(
            self.transform_frame,
            text="Output Size: 0 × 0 px | Zoom: 100%",
            font=("Segoe UI", 8),
            bg=self.card_color,
            fg=self.secondary_text
        )
        crop_info_label._imagelab_crop_info = True
        crop_info_label.grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="w",
            padx=5,
            pady=(0, 3)
        )

        tk.Label(
            self.transform_frame,
            text="Mouse wheel: zoom image • Drag image: reposition • Final image size stays unchanged",
            font=("Segoe UI", 8),
            bg=self.card_color,
            fg=self.secondary_text
        ).grid(
            row=2,
            column=0,
            columnspan=4,
            sticky="w",
            padx=5,
            pady=(0, 4)
        )

        ttk.Button(
            self.transform_frame,
            text="Apply Crop",
            command=self.apply_inline_crop
        ).grid(
            row=3,
            column=0,
            padx=4,
            pady=4,
            sticky="ew"
        )

        ttk.Button(
            self.transform_frame,
            text="Reset Crop",
            command=self.reset_inline_crop
        ).grid(
            row=3,
            column=1,
            padx=4,
            pady=4,
            sticky="ew"
        )

        ttk.Button(
            self.transform_frame,
            text="Cancel Crop",
            command=self.cancel_inline_crop
        ).grid(
            row=3,
            column=2,
            padx=4,
            pady=4,
            sticky="ew"
        )

        ttk.Button(
            self.transform_frame,
            text="Back to Transform",
            command=self.cancel_inline_crop
        ).grid(
            row=3,
            column=3,
            padx=4,
            pady=4,
            sticky="ew"
        )

        for column in range(4):
            self.transform_frame.grid_columnconfigure(
                column,
                weight=1
            )

        self.status.config(
            text="Crop mode: use the mouse wheel to zoom and drag to reposition"
        )

    def _crop_geometry(self):

        if self.crop_canvas is None or self.base_image is None:
            return None

        canvas_width = self.crop_canvas.winfo_width()
        canvas_height = self.crop_canvas.winfo_height()

        if canvas_width <= 20 or canvas_height <= 20:
            return None

        available_width = max(canvas_width - 30, 100)
        available_height = max(canvas_height - 30, 100)

        image_aspect = self.base_image.width / self.base_image.height

        fit_width = available_width
        fit_height = fit_width / image_aspect

        if fit_height > available_height:
            fit_height = available_height
            fit_width = fit_height * image_aspect

        # Keep the crop box under the user's control.  It is initialized once
        # at 82% of the fitted image, then remains where the user moves/resizes
        # it while the image itself is zoomed with the mouse wheel.
        if self.crop_frame is None:
            frame_width = fit_width * 0.82
            frame_height = fit_height * 0.82

            frame_left = (canvas_width - frame_width) / 2
            frame_top = (canvas_height - frame_height) / 2
            frame_right = frame_left + frame_width
            frame_bottom = frame_top + frame_height

            self.crop_frame = (
                frame_left,
                frame_top,
                frame_right,
                frame_bottom
            )
        else:
            frame_left, frame_top, frame_right, frame_bottom = self.crop_frame

            # Keep the existing crop box inside the resized canvas.
            frame_width = min(
                max(self.crop_min_size, frame_right - frame_left),
                canvas_width - 10
            )
            frame_height = min(
                max(self.crop_min_size, frame_bottom - frame_top),
                canvas_height - 10
            )

            frame_left = min(max(frame_left, 5), canvas_width - frame_width - 5)
            frame_top = min(max(frame_top, 5), canvas_height - frame_height - 5)
            frame_right = frame_left + frame_width
            frame_bottom = frame_top + frame_height

            self.crop_frame = (
                frame_left,
                frame_top,
                frame_right,
                frame_bottom
            )

        return (
            canvas_width,
            canvas_height,
            fit_width,
            fit_height,
            frame_left,
            frame_top,
            frame_right,
            frame_bottom
        )

    def _clamp_crop_pan(self, display_width, display_height, frame):

        if self.crop_canvas is None:
            return

        canvas_width = self.crop_canvas.winfo_width()
        canvas_height = self.crop_canvas.winfo_height()
        frame_left, frame_top, frame_right, frame_bottom = frame

        image_left = (
            (canvas_width - display_width) / 2
            + self.crop_pan_x
        )
        image_top = (
            (canvas_height - display_height) / 2
            + self.crop_pan_y
        )

        # Keep the image covering the complete crop frame.
        min_left = frame_right - display_width
        max_left = frame_left
        min_top = frame_bottom - display_height
        max_top = frame_top

        if display_width >= (frame_right - frame_left):
            image_left = min(max(image_left, min_left), max_left)
        else:
            image_left = (frame_left + frame_right - display_width) / 2

        if display_height >= (frame_bottom - frame_top):
            image_top = min(max(image_top, min_top), max_top)
        else:
            image_top = (frame_top + frame_bottom - display_height) / 2

        self.crop_pan_x = (
            image_left
            - (canvas_width - display_width) / 2
        )
        self.crop_pan_y = (
            image_top
            - (canvas_height - display_height) / 2
        )

    def _redraw_crop_editor(self):

        if not self.crop_mode:
            return

        if self.crop_canvas is None or self.base_image is None:
            return

        geometry = self._crop_geometry()
        if geometry is None:
            return

        (
            canvas_width,
            canvas_height,
            fit_width,
            fit_height,
            frame_left,
            frame_top,
            frame_right,
            frame_bottom
        ) = geometry

        display_width = max(
            1,
            int(fit_width * self.crop_zoom)
        )
        display_height = max(
            1,
            int(fit_height * self.crop_zoom)
        )

        self._clamp_crop_pan(
            display_width,
            display_height,
            self.crop_frame
        )

        image_left = (
            (canvas_width - display_width) / 2
            + self.crop_pan_x
        )
        image_top = (
            (canvas_height - display_height) / 2
            + self.crop_pan_y
        )
        image_right = image_left + display_width
        image_bottom = image_top + display_height

        self.crop_image_bounds = (
            image_left,
            image_top,
            image_right,
            image_bottom
        )

        preview = self.base_image.resize(
            (display_width, display_height),
            Image.Resampling.BILINEAR
        )

        self.crop_image_photo = ImageTk.PhotoImage(preview)
        self.crop_canvas.delete("all")

        self.crop_canvas.create_image(
            image_left,
            image_top,
            image=self.crop_image_photo,
            anchor="nw",
            tags="crop_image"
        )

        # Darken everything outside the fixed crop frame.
        self.crop_canvas.create_rectangle(
            0,
            0,
            canvas_width,
            frame_top,
            fill="#000000",
            stipple="gray50",
            outline=""
        )
        self.crop_canvas.create_rectangle(
            0,
            frame_bottom,
            canvas_width,
            canvas_height,
            fill="#000000",
            stipple="gray50",
            outline=""
        )
        self.crop_canvas.create_rectangle(
            0,
            frame_top,
            frame_left,
            frame_bottom,
            fill="#000000",
            stipple="gray50",
            outline=""
        )
        self.crop_canvas.create_rectangle(
            frame_right,
            frame_top,
            canvas_width,
            frame_bottom,
            fill="#000000",
            stipple="gray50",
            outline=""
        )

        # Fixed-in-place crop box that the user can move and resize.
        self.crop_rect_id = self.crop_canvas.create_rectangle(
            frame_left,
            frame_top,
            frame_right,
            frame_bottom,
            outline="white",
            width=2,
            tags="crop_rect"
        )

        # Rule-of-thirds grid.
        for fraction in (1 / 3, 2 / 3):
            x = frame_left + (frame_right - frame_left) * fraction
            y = frame_top + (frame_bottom - frame_top) * fraction

            self.crop_canvas.create_line(
                x, frame_top, x, frame_bottom,
                fill="white", dash=(4, 4), tags="crop_grid"
            )
            self.crop_canvas.create_line(
                frame_left, y, frame_right, y,
                fill="white", dash=(4, 4), tags="crop_grid"
            )

        # Four draggable corner handles restore the original crop-box control.
        self.crop_handles = {}
        handle_size = self.crop_handle_size
        positions = {
            "nw": (frame_left, frame_top),
            "ne": (frame_right, frame_top),
            "sw": (frame_left, frame_bottom),
            "se": (frame_right, frame_bottom)
        }

        for name, (x, y) in positions.items():
            self.crop_handles[name] = self.crop_canvas.create_rectangle(
                x - handle_size, y - handle_size,
                x + handle_size, y + handle_size,
                fill="white",
                outline="black",
                width=1,
                tags=("crop_handle", f"crop_handle_{name}")
            )

        self._update_crop_info()


    def _crop_mouse_wheel(self, event):

        if not self.crop_mode:
            return "break"

        if getattr(event, "delta", 0) > 0:
            direction = 1
        elif getattr(event, "delta", 0) < 0:
            direction = -1
        else:
            direction = 0

        if direction:
            self._change_crop_zoom(direction, event.x, event.y)

        return "break"

    def _change_crop_zoom(self, direction, mouse_x=None, mouse_y=None):

        if not self.crop_mode or self.base_image is None:
            return

        old_zoom = self.crop_zoom

        if direction > 0:
            new_zoom = min(4.0, old_zoom * 1.10)
        else:
            new_zoom = max(1.0, old_zoom / 1.10)

        if abs(new_zoom - old_zoom) < 0.0001:
            return

        # Keep the point under the mouse in approximately the same place
        # while zooming, like a phone/gallery image editor.
        if (
            mouse_x is not None
            and mouse_y is not None
            and self.crop_image_bounds is not None
            and self.crop_canvas is not None
        ):
            old_left, old_top, _, _ = self.crop_image_bounds

            source_x = (
                mouse_x - old_left
            ) / max(1, self.crop_scale)
            source_y = (
                mouse_y - old_top
            ) / max(1, self.crop_scale)

            old_display_width = max(
                1,
                self.crop_image_bounds[2] - self.crop_image_bounds[0]
            )
            old_display_height = max(
                1,
                self.crop_image_bounds[3] - self.crop_image_bounds[1]
            )

            relative_x = (
                mouse_x - old_left
            ) / old_display_width
            relative_y = (
                mouse_y - old_top
            ) / old_display_height

            geometry = self._crop_geometry()
            if geometry is not None:
                canvas_width, canvas_height, fit_width, fit_height, *_ = geometry
                new_width = fit_width * new_zoom
                new_height = fit_height * new_zoom

                desired_left = mouse_x - relative_x * new_width
                desired_top = mouse_y - relative_y * new_height

                self.crop_pan_x = (
                    desired_left
                    - (canvas_width - new_width) / 2
                )
                self.crop_pan_y = (
                    desired_top
                    - (canvas_height - new_height) / 2
                )

        self.crop_zoom = new_zoom
        self._redraw_crop_editor()

        self.status.config(
            text=f"Crop zoom: {int(round(self.crop_zoom * 100))}%"
        )

    def _crop_hit_handle(self, x, y):

        if not self.crop_frame:
            return None

        handle_size = self.crop_handle_size + 4
        positions = {
            "nw": (self.crop_frame[0], self.crop_frame[1]),
            "ne": (self.crop_frame[2], self.crop_frame[1]),
            "sw": (self.crop_frame[0], self.crop_frame[3]),
            "se": (self.crop_frame[2], self.crop_frame[3])
        }

        for name, (hx, hy) in positions.items():
            if abs(x - hx) <= handle_size and abs(y - hy) <= handle_size:
                return name

        return None

    def _start_crop_interaction(self, event):

        if not self.crop_mode or self.crop_frame is None:
            return

        handle = self._crop_hit_handle(event.x, event.y)

        if handle is not None:
            mode = handle
        else:
            left, top, right, bottom = self.crop_frame
            if left <= event.x <= right and top <= event.y <= bottom:
                mode = "move_frame"
            else:
                mode = "move_image"

        self.crop_drag_data = {
            "mode": mode,
            "start_x": event.x,
            "start_y": event.y,
            "start_frame": tuple(self.crop_frame),
            "start_pan_x": self.crop_pan_x,
            "start_pan_y": self.crop_pan_y
        }

    def _drag_crop_interaction(self, event):

        if not self.crop_mode or not self.crop_drag_data:
            return

        mode = self.crop_drag_data["mode"]
        dx = event.x - self.crop_drag_data["start_x"]
        dy = event.y - self.crop_drag_data["start_y"]

        if mode == "move_image":
            self.crop_pan_x = self.crop_drag_data["start_pan_x"] + dx
            self.crop_pan_y = self.crop_drag_data["start_pan_y"] + dy
            self._redraw_crop_editor()
            return

        left, top, right, bottom = self.crop_drag_data["start_frame"]
        canvas_width = self.crop_canvas.winfo_width()
        canvas_height = self.crop_canvas.winfo_height()
        min_size = self.crop_min_size

        if mode == "move_frame":
            width = right - left
            height = bottom - top
            new_left = left + dx
            new_top = top + dy

            new_left = min(max(5, new_left), canvas_width - width - 5)
            new_top = min(max(5, new_top), canvas_height - height - 5)

            self.crop_frame = (
                new_left, new_top,
                new_left + width, new_top + height
            )

        else:
            new_left, new_top, new_right, new_bottom = left, top, right, bottom

            if "w" in mode:
                new_left = min(event.x, right - min_size)
                new_left = max(5, new_left)
            if "e" in mode:
                new_right = max(event.x, left + min_size)
                new_right = min(canvas_width - 5, new_right)
            if "n" in mode:
                new_top = min(event.y, bottom - min_size)
                new_top = max(5, new_top)
            if "s" in mode:
                new_bottom = max(event.y, top + min_size)
                new_bottom = min(canvas_height - 5, new_bottom)

            self.crop_frame = (
                new_left, new_top, new_right, new_bottom
            )

        self._redraw_crop_editor()

    def _stop_crop_interaction(self, event=None):
        self.crop_drag_data = {}

    # Backward-compatible names used by older crop code.
    def _start_crop_image_drag(self, event):
        self._start_crop_interaction(event)

    def _drag_crop_image(self, event):
        self._drag_crop_interaction(event)

    def _stop_crop_image_drag(self, event=None):
        self._stop_crop_interaction(event)

    def _update_crop_info(self):

        if not self.crop_mode or self.base_image is None:
            return

        if self.crop_canvas is None or self.crop_frame is None:
            return

        geometry = self._crop_geometry()
        if geometry is None:
            return

        (
            _,
            _,
            fit_width,
            fit_height,
            frame_left,
            frame_top,
            frame_right,
            frame_bottom
        ) = geometry

        if self.crop_image_bounds is None:
            return

        image_left, image_top, _, _ = self.crop_image_bounds

        source_scale_x = fit_width * self.crop_zoom / self.base_image.width
        source_scale_y = fit_height * self.crop_zoom / self.base_image.height

        source_left = int(
            round((frame_left - image_left) / max(source_scale_x, 1e-9))
        )
        source_top = int(
            round((frame_top - image_top) / max(source_scale_y, 1e-9))
        )
        source_right = int(
            round((frame_right - image_left) / max(source_scale_x, 1e-9))
        )
        source_bottom = int(
            round((frame_bottom - image_top) / max(source_scale_y, 1e-9))
        )

        source_left = max(0, min(self.base_image.width - 1, source_left))
        source_top = max(0, min(self.base_image.height - 1, source_top))
        source_right = max(source_left + 1, min(self.base_image.width, source_right))
        source_bottom = max(source_top + 1, min(self.base_image.height, source_bottom))

        crop_width = source_right - source_left
        crop_height = source_bottom - source_top

        if self.transform_frame is not None:
            for widget in self.transform_frame.winfo_children():
                if getattr(widget, "_imagelab_crop_info", False):
                    widget.config(
                        text=(
                            f"Output Size: {self.base_image.width} × "
                            f"{self.base_image.height} px  |  "
                            f"Source Area: {crop_width} × {crop_height} px  |  "
                            f"Zoom: {int(round(self.crop_zoom * 100))}%"
                        )
                    )
                    return

    def reset_inline_crop(self):

        if not self.crop_mode or self.base_image is None:
            return

        self.crop_zoom = 1.0
        self.crop_pan_x = 0.0
        self.crop_pan_y = 0.0
        self.crop_drag_data = {}
        self.crop_frame = None

        self._redraw_crop_editor()
        self.status.config(
            text="Crop view reset to 100%"
        )

    def _get_inline_crop_source_box(self):

        if (
            not self.crop_mode
            or self.base_image is None
            or self.crop_frame is None
            or self.crop_image_bounds is None
        ):
            return None

        (
            _,
            _,
            fit_width,
            fit_height,
            frame_left,
            frame_top,
            frame_right,
            frame_bottom
        ) = self._crop_geometry()

        image_left, image_top, _, _ = self.crop_image_bounds

        display_scale_x = (
            fit_width * self.crop_zoom
            / self.base_image.width
        )
        display_scale_y = (
            fit_height * self.crop_zoom
            / self.base_image.height
        )

        if display_scale_x <= 0 or display_scale_y <= 0:
            return None

        x1 = int(
            round((frame_left - image_left) / display_scale_x)
        )
        y1 = int(
            round((frame_top - image_top) / display_scale_y)
        )
        x2 = int(
            round((frame_right - image_left) / display_scale_x)
        )
        y2 = int(
            round((frame_bottom - image_top) / display_scale_y)
        )

        x1 = max(0, min(self.base_image.width - 1, x1))
        y1 = max(0, min(self.base_image.height - 1, y1))
        x2 = max(x1 + 1, min(self.base_image.width, x2))
        y2 = max(y1 + 1, min(self.base_image.height, y2))

        return x1, y1, x2, y2

    def apply_inline_crop(self):

        if not self.crop_mode or self.base_image is None:
            return

        crop_box = self._get_inline_crop_source_box()

        if crop_box is None:
            messagebox.showwarning(
                "Invalid Crop",
                "Please select a valid crop area."
            )
            return

        x1, y1, x2, y2 = crop_box

        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning(
                "Invalid Crop",
                "Please select a valid crop area."
            )
            return

        cropped = self.base_image.crop(
            (x1, y1, x2, y2)
        )

        # Keep the final output exactly the same pixel dimensions as the
        # image before cropping. This is the key phone/gallery-style behavior:
        # zoom and reposition decide what is visible, while the final canvas
        # size remains unchanged.
        final_size = (
            self.base_image.width,
            self.base_image.height
        )

        if cropped.size != final_size:
            cropped = cropped.resize(
                final_size,
                Image.Resampling.LANCZOS
            )

        self.end_inline_crop()

        self.apply_transform_result(
            cropped,
            (
                f"Crop applied "
                f"({cropped.width} × {cropped.height} px)"
            )
        )

        self.status.config(
            text=(
                f"Crop applied | Final size unchanged: "
                f"{cropped.width} × {cropped.height} px"
            )
        )

    def cancel_inline_crop(self):

        if not self.crop_mode:
            return

        self.end_inline_crop()
        self.show_transform_control()
        self.status.config(
            text="Crop cancelled"
        )

    def end_inline_crop(self):

        self.crop_mode = False
        self.crop_drag_data = {}

        if self.crop_canvas is not None:
            try:
                self.crop_canvas.destroy()
            except tk.TclError:
                pass

        self.crop_canvas = None
        self.crop_image_photo = None
        self.crop_rect_id = None
        self.crop_handles = {}
        self.crop_frame = None
        self.crop_image_bounds = None
        self.crop_zoom = 1.0
        self.crop_pan_x = 0.0
        self.crop_pan_y = 0.0

        self.preview_label.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(5, 10)
        )

        self.display_image_on_preview()

    # ============================================================
    # BEFORE / AFTER
    # ============================================================

    def show_before_after(self):

        if self.original_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        self.commit_slider_history()

        if self.comparison_window is not None:

            try:

                if self.comparison_window.winfo_exists():

                    self.update_comparison_window()
                    self.comparison_window.lift()

                    return

            except tk.TclError:
                pass

        self.comparison_mode = "Side by Side"
        self.comparison_zoom = 1.0
        self.comparison_split_position = 0.5
        self.comparison_dragging_split = False

        self.comparison_window = tk.Toplevel(
            self.root
        )

        self.comparison_window.title(
            "ImageLab - Before / After"
        )

        self.comparison_window.geometry(
            "1150x720"
        )

        self.comparison_window.minsize(
            900,
            600
        )

        self.comparison_window.configure(
            bg=self.bg_color
        )

        self.comparison_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_comparison_window
        )

        # --------------------------------------------------------
        # HEADER
        # --------------------------------------------------------

        header = tk.Frame(
            self.comparison_window,
            bg=self.card_color,
            height=65
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        tk.Label(
            header,
            text="Before / After Comparison",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            side="left",
            padx=20
        )

        controls = tk.Frame(
            header,
            bg=self.card_color
        )

        controls.pack(
            side="right",
            padx=15
        )

        tk.Label(
            controls,
            text="View:",
            font=("Segoe UI", 9, "bold"),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            side="left",
            padx=(0, 5)
        )

        self.comparison_mode_var = tk.StringVar(
            value="Side by Side"
        )

        mode_menu = ttk.Combobox(
            controls,
            textvariable=self.comparison_mode_var,
            values=[
                "Side by Side",
                "Split View"
            ],
            state="readonly",
            width=14
        )

        mode_menu.pack(
            side="left",
            padx=5
        )

        mode_menu.bind(
            "<<ComboboxSelected>>",
            lambda event: self.change_comparison_mode()
        )

        ttk.Button(
            controls,
            text="−",
            width=3,
            command=self.comparison_zoom_out
        ).pack(
            side="left",
            padx=(10, 2)
        )

        self.comparison_zoom_label = tk.Label(
            controls,
            text="100%",
            width=7,
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        )

        self.comparison_zoom_label.pack(
            side="left"
        )

        ttk.Button(
            controls,
            text="+",
            width=3,
            command=self.comparison_zoom_in
        ).pack(
            side="left",
            padx=2
        )

        ttk.Button(
            controls,
            text="100%",
            command=self.comparison_zoom_100
        ).pack(
            side="left",
            padx=4
        )

        ttk.Button(
            controls,
            text="Refresh",
            command=self.update_comparison_window
        ).pack(
            side="left",
            padx=(4, 0)
        )

        # --------------------------------------------------------
        # INFO BAR
        # --------------------------------------------------------

        info_bar = tk.Frame(
            self.comparison_window,
            bg=self.bg_color,
            height=32
        )

        info_bar.pack(
            fill="x",
            padx=15,
            pady=(8, 0)
        )

        info_bar.pack_propagate(False)

        self.comparison_info_label = tk.Label(
            info_bar,
            text=(
                "Side-by-side comparison • "
                "Zoom is synchronized"
            ),
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg=self.secondary_text
        )

        self.comparison_info_label.pack(
            anchor="w"
        )

        # --------------------------------------------------------
        # COMPARISON AREA
        # --------------------------------------------------------

        self.comparison_area = tk.Frame(
            self.comparison_window,
            bg=self.bg_color
        )

        self.comparison_area.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        self.comparison_area.bind(
            "<Configure>",
            lambda event: self.update_comparison_window()
        )

        # Canvas is used for both comparison modes.
        self.comparison_canvas = tk.Canvas(
            self.comparison_area,
            bg=self.card_color,
            highlightthickness=1,
            highlightbackground="#d1d5db"
        )

        self.comparison_canvas.pack(
            fill="both",
            expand=True
        )

        self.comparison_canvas.bind(
            "<MouseWheel>",
            self.handle_comparison_mouse_zoom
        )

        self.comparison_canvas.bind(
            "<Button-4>",
            self.handle_comparison_mouse_zoom
        )

        self.comparison_canvas.bind(
            "<Button-5>",
            self.handle_comparison_mouse_zoom
        )

        self.comparison_canvas.bind(
            "<ButtonPress-1>",
            self.start_comparison_split_drag
        )

        self.comparison_canvas.bind(
            "<B1-Motion>",
            self.drag_comparison_split
        )

        self.comparison_canvas.bind(
            "<ButtonRelease-1>",
            self.stop_comparison_split_drag
        )

        # --------------------------------------------------------
        # FOOTER
        # --------------------------------------------------------

        footer = tk.Frame(
            self.comparison_window,
            bg=self.card_color,
            height=38
        )

        footer.pack(
            fill="x"
        )

        footer.pack_propagate(False)

        tk.Label(
            footer,
            text=(
                "Split View: drag the center divider • "
                "Mouse wheel: synchronized zoom"
            ),
            font=("Segoe UI", 8),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            anchor="center",
            pady=9
        )

        self.update_comparison_window()

    def change_comparison_mode(self):

        if self.comparison_window is None:
            return

        try:

            if not self.comparison_window.winfo_exists():
                return

        except tk.TclError:
            return

        self.comparison_mode = (
            self.comparison_mode_var.get()
        )

        if self.comparison_mode == "Split View":

            self.comparison_info_label.config(
                text=(
                    "Split-view comparison • "
                    "Drag the center divider • "
                    "Zoom is synchronized"
                )
            )

        else:

            self.comparison_info_label.config(
                text=(
                    "Side-by-side comparison • "
                    "Zoom is synchronized"
                )
            )

        self.update_comparison_window()

    def comparison_zoom_in(self):

        if self.comparison_window is None:
            return

        self.comparison_zoom = min(
            3.0,
            self.comparison_zoom + 0.1
        )

        self.update_comparison_window()

    def comparison_zoom_out(self):

        if self.comparison_window is None:
            return

        self.comparison_zoom = max(
            0.2,
            self.comparison_zoom - 0.1
        )

        self.update_comparison_window()

    def comparison_zoom_100(self):

        if self.comparison_window is None:
            return

        self.comparison_zoom = 1.0

        self.update_comparison_window()

    def handle_comparison_mouse_zoom(self, event):

        if self.comparison_window is None:
            return

        if getattr(event, "delta", 0) > 0:

            self.comparison_zoom_in()

        elif getattr(event, "delta", 0) < 0:

            self.comparison_zoom_out()

        elif getattr(event, "num", None) == 4:

            self.comparison_zoom_in()

        elif getattr(event, "num", None) == 5:

            self.comparison_zoom_out()

    def start_comparison_split_drag(self, event):

        if self.comparison_mode != "Split View":
            return

        if self.comparison_canvas is None:
            return

        canvas_width = self.comparison_canvas.winfo_width()

        if canvas_width <= 1:
            return

        divider_x = (
            canvas_width
            * self.comparison_split_position
        )

        if abs(event.x - divider_x) <= 25:

            self.comparison_dragging_split = True

    def drag_comparison_split(self, event):

        if not self.comparison_dragging_split:
            return

        if self.comparison_canvas is None:
            return

        canvas_width = self.comparison_canvas.winfo_width()

        if canvas_width <= 1:
            return

        self.comparison_split_position = max(
            0.05,
            min(
                0.95,
                event.x / canvas_width
            )
        )

        self.update_comparison_window()

    def stop_comparison_split_drag(self, event):

        self.comparison_dragging_split = False

    def _comparison_prepare_image(
        self,
        image,
        max_width,
        max_height
    ):

        image = image.copy()

        if image.mode not in (
            "RGB",
            "RGBA",
            "L"
        ):

            image = image.convert(
                "RGB"
            )

        if image.mode == "L":

            image = image.convert(
                "RGB"
            )

        if max_width <= 1 or max_height <= 1:
            return image

        fit_ratio = min(
            max_width / image.width,
            max_height / image.height
        )

        scale = (
            fit_ratio
            * self.comparison_zoom
        )

        scale = max(
            0.01,
            scale
        )

        new_width = max(
            1,
            int(image.width * scale)
        )

        new_height = max(
            1,
            int(image.height * scale)
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        return image

    def _comparison_create_photo(
        self,
        image
    ):

        return ImageTk.PhotoImage(
            image
        )

    def _comparison_draw_card(
        self,
        canvas,
        x1,
        y1,
        x2,
        y2,
        title,
        image,
        title_fg
    ):

        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=self.card_color,
            outline="#d1d5db"
        )

        title_y = y1 + 25

        canvas.create_text(
            (x1 + x2) / 2,
            title_y,
            text=title,
            font=("Segoe UI", 11, "bold"),
            fill=title_fg
        )

        image_area_top = y1 + 48

        area_width = max(
            1,
            x2 - x1 - 20
        )

        area_height = max(
            1,
            y2 - image_area_top - 15
        )

        prepared = self._comparison_prepare_image(
            image,
            area_width,
            area_height
        )

        photo = self._comparison_create_photo(
            prepared
        )

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            image_area_top + y2
        ) / 2

        canvas.create_image(
            center_x,
            center_y,
            image=photo,
            anchor="center"
        )

        return photo

    def _comparison_draw_side_by_side(self):

        canvas = self.comparison_canvas

        width = max(
            1,
            canvas.winfo_width()
        )

        height = max(
            1,
            canvas.winfo_height()
        )

        gap = 8
        card_width = (
            width - gap
        ) / 2

        before_photo = self._comparison_draw_card(
            canvas,
            0,
            0,
            card_width,
            height,
            "BEFORE — ORIGINAL",
            self.original_image,
            self.secondary_text
        )

        after_photo = self._comparison_draw_card(
            canvas,
            card_width + gap,
            0,
            width,
            height,
            "AFTER — EDITED",
            self.current_image,
            self.accent_color
        )

        self.comparison_before_photo = (
            before_photo
        )

        self.comparison_after_photo = (
            after_photo
        )

    def _comparison_draw_split_view(self):

        canvas = self.comparison_canvas

        width = max(
            1,
            canvas.winfo_width()
        )

        height = max(
            1,
            canvas.winfo_height()
        )

        # Fit both images to the complete comparison area.
        before = self._comparison_prepare_image(
            self.original_image,
            width - 20,
            height - 70
        )

        after = self._comparison_prepare_image(
            self.current_image,
            width - 20,
            height - 70
        )

        # Use the same scale for both images so the comparison
        # remains visually synchronized.
        original_w = max(
            1,
            self.original_image.width
        )

        original_h = max(
            1,
            self.original_image.height
        )

        current_w = max(
            1,
            self.current_image.width
        )

        current_h = max(
            1,
            self.current_image.height
        )

        fit_before = min(
            (width - 20) / original_w,
            (height - 70) / original_h
        )

        fit_after = min(
            (width - 20) / current_w,
            (height - 70) / current_h
        )

        shared_scale = min(
            fit_before,
            fit_after
        )

        shared_scale = max(
            0.01,
            shared_scale
        )

        shared_scale *= self.comparison_zoom

        before = self.original_image.copy()

        before = before.resize(
            (
                max(1, int(original_w * shared_scale)),
                max(1, int(original_h * shared_scale))
            ),
            Image.Resampling.LANCZOS
        )

        after = self.current_image.copy()

        after = after.resize(
            (
                max(1, int(current_w * shared_scale)),
                max(1, int(current_h * shared_scale))
            ),
            Image.Resampling.LANCZOS
        )

        # The split view is based on a common centered image
        # area. This keeps the divider meaningful even when
        # the before and after dimensions differ.
        image_width = max(
            before.width,
            after.width
        )

        image_height = max(
            before.height,
            after.height
        )

        canvas_center_x = width / 2
        canvas_center_y = (
            height + 45
        ) / 2

        image_left = (
            canvas_center_x
            - image_width / 2
        )

        image_top = (
            canvas_center_y
            - image_height / 2
        )

        # Background area.
        canvas.create_rectangle(
            0,
            0,
            width,
            height,
            fill=self.card_color,
            outline=""
        )

        # Labels.
        canvas.create_text(
            width * 0.25,
            20,
            text="BEFORE — ORIGINAL",
            font=("Segoe UI", 11, "bold"),
            fill=self.secondary_text
        )

        canvas.create_text(
            width * 0.75,
            20,
            text="AFTER — EDITED",
            font=("Segoe UI", 11, "bold"),
            fill=self.accent_color
        )

        before_photo = ImageTk.PhotoImage(
            before
        )

        after_photo = ImageTk.PhotoImage(
            after
        )

        self.comparison_before_photo = (
            before_photo
        )

        self.comparison_after_photo = (
            after_photo
        )

        canvas.create_image(
            canvas_center_x,
            canvas_center_y,
            image=before_photo,
            anchor="center"
        )

        canvas.create_image(
            canvas_center_x,
            canvas_center_y,
            image=after_photo,
            anchor="center"
        )

        # Clip the after image visually by drawing the left side
        # back over it with the before image. A canvas clip is not
        # available in Tkinter, so we use cropped image segments.
        split_x = (
            width
            * self.comparison_split_position
        )

        # Redraw with correctly cropped halves.
        canvas.delete("all")

        canvas.create_rectangle(
            0,
            0,
            width,
            height,
            fill=self.card_color,
            outline=""
        )

        # Re-create centered images and crop them into two halves.
        before_crop = before.copy()

        after_crop = after.copy()

        half_left = max(
            0,
            int(
                split_x
                - image_left
            )
        )

        half_left = min(
            before_crop.width,
            half_left
        )

        half_right = max(
            0,
            int(
                split_x
                - image_left
            )
        )

        half_right = min(
            after_crop.width,
            half_right
        )

        before_left = before_crop.crop(
            (
                0,
                0,
                half_left,
                before_crop.height
            )
        )

        after_right = after_crop.crop(
            (
                half_right,
                0,
                after_crop.width,
                after_crop.height
            )
        )

        if before_left.width > 0:

            before_left_photo = ImageTk.PhotoImage(
                before_left
            )

            self.comparison_before_region = (
                before_left_photo
            )

            canvas.create_image(
                image_left,
                image_top,
                image=before_left_photo,
                anchor="nw"
            )

        if after_right.width > 0:

            after_right_photo = ImageTk.PhotoImage(
                after_right
            )

            self.comparison_after_region = (
                after_right_photo
            )

            canvas.create_image(
                image_left + half_right,
                image_top,
                image=after_right_photo,
                anchor="nw"
            )

        # Divider.
        canvas.create_line(
            split_x,
            0,
            split_x,
            height,
            fill=self.accent_color,
            width=3
        )

        canvas.create_rectangle(
            split_x - 24,
            height / 2 - 18,
            split_x + 24,
            height / 2 + 18,
            fill=self.card_color,
            outline=self.accent_color,
            width=2
        )

        canvas.create_text(
            split_x,
            height / 2,
            text="↔",
            font=("Segoe UI", 14, "bold"),
            fill=self.accent_color
        )

        canvas.create_text(
            width / 2,
            height - 18,
            text="Drag divider to compare",
            font=("Segoe UI", 8),
            fill=self.secondary_text
        )

        self.comparison_before_photo = before_left_photo if before_left.width > 0 else None
        self.comparison_after_photo = after_right_photo if after_right.width > 0 else None

    def update_comparison_window(self):

        if self.comparison_window is None:
            return

        try:

            if not self.comparison_window.winfo_exists():
                return

        except tk.TclError:
            return

        if self.original_image is None:
            return

        if self.current_image is None:
            return

        if self.comparison_canvas is None:
            return

        try:

            self.comparison_canvas.delete(
                "all"
            )

            if self.comparison_mode == "Split View":

                self._comparison_draw_split_view()

            else:

                self._comparison_draw_side_by_side()

            self.comparison_zoom_label.config(
                text=f"{self.comparison_zoom * 100:.0f}%"
            )

        except tk.TclError:
            return

    def close_comparison_window(self):

        try:

            self.comparison_window.destroy()

        except (
            tk.TclError,
            AttributeError
        ):
            pass

        self.comparison_window = None
        self.comparison_canvas = None
        self.comparison_before_photo = None
        self.comparison_after_photo = None
        self.comparison_before_region = None
        self.comparison_after_region = None
        self.comparison_dragging_split = False

    # ============================================================
    # PROCESSING HISTORY
    # ============================================================

    def show_history(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        if self.history_window is not None:

            try:

                if self.history_window.winfo_exists():

                    self.update_history_window()

                    self.history_window.lift()

                    return

            except tk.TclError:
                pass

        self.history_window = tk.Toplevel(
            self.root
        )

        self.history_window.title(
            "ImageLab - Processing History"
        )

        self.history_window.geometry(
            "600x500"
        )

        self.history_window.configure(
            bg=self.bg_color
        )

        self.history_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_history_window
        )

        tk.Label(
            self.history_window,
            text="Processing History",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(
            pady=(20, 5)
        )

        tk.Label(
            self.history_window,
            text="Recent editing operations",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg=self.secondary_text
        ).pack(
            pady=(0, 15)
        )

        list_frame = tk.Frame(
            self.history_window,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )

        list_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        self.history_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
            bg=self.card_color,
            fg=self.text_color,
            relief="flat",
            borderwidth=0
        )

        self.history_listbox.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        buttons = tk.Frame(
            self.history_window,
            bg=self.bg_color
        )

        buttons.pack(
            pady=10
        )

        ttk.Button(
            buttons,
            text="Clear History",
            command=self.clear_history
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Close",
            command=self.close_history_window
        ).pack(
            side="left",
            padx=5
        )

        self.update_history_window()

    def update_history_window(self):

        if self.history_window is None:
            return

        try:

            if not self.history_window.winfo_exists():
                return

        except tk.TclError:
            return

        if self.history_listbox is None:
            return

        self.history_listbox.delete(
            0,
            tk.END
        )

        if not self.history:

            self.history_listbox.insert(
                tk.END,
                "No processing operations yet."
            )

            return

        for index, operation in enumerate(
            self.history,
            start=1
        ):

            self.history_listbox.insert(
                tk.END,
                f"{index}. {operation}"
            )

    def clear_history(self):

        if getattr(self, "settings_confirm_reset", True):
            if not messagebox.askyesno(
                "Clear History",
                "Clear Undo, Redo, Processing History, and Processing Stack?"
            ):
                return

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.history.clear()
        self.processing_layers.clear()
        self.update_processing_stack_window()

        self.update_history_window()

        self.status.config(
            text="Processing history cleared"
        )

    def close_history_window(self):

        try:

            self.history_window.destroy()

        except (
            tk.TclError,
            AttributeError
        ):
            pass

        self.history_window = None
        self.history_listbox = None

    # ============================================================
    # IMAGE ANALYSIS
    # ============================================================

    def show_analysis(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return
        self.set_active_tool("analyze")


        self.commit_slider_history()

        if self.analysis_window is not None:

            try:

                if self.analysis_window.winfo_exists():

                    self.update_analysis_window()

                    self.analysis_window.lift()

                    return

            except tk.TclError:
                pass

        self.analysis_window = tk.Toplevel(
            self.root
        )

        self.analysis_window.title(
            "ImageLab - Image Analysis"
        )

        self.analysis_window.geometry(
            "1100x700"
        )

        self.analysis_window.minsize(
            900,
            600
        )

        self.analysis_window.configure(
            bg=self.bg_color
        )

        self.analysis_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_analysis_window
        )

        header = tk.Frame(
            self.analysis_window,
            bg=self.card_color,
            height=70
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        tk.Label(
            header,
            text="Image Analysis",
            font=("Segoe UI", 20, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            side="left",
            padx=20
        )

        ttk.Button(
            header,
            text="Refresh Analysis",
            command=self.update_analysis_window
        ).pack(
            side="right",
            padx=20
        )

        content = tk.Frame(
            self.analysis_window,
            bg=self.bg_color
        )

        content.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        content.grid_columnconfigure(
            0,
            weight=0
        )

        content.grid_columnconfigure(
            1,
            weight=1
        )

        content.grid_rowconfigure(
            0,
            weight=1
        )

        self.analysis_stats_frame = tk.Frame(
            content,
            bg=self.card_color,
            bd=1,
            relief="solid",
            width=370
        )

        self.analysis_stats_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        self.analysis_stats_frame.grid_propagate(
            False
        )

        histogram_card = tk.Frame(
            content,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )

        histogram_card.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        tk.Label(
            histogram_card,
            text="RGB Histogram",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(10, 0)
        )

        tk.Label(
            histogram_card,
            text="Pixel intensity distribution (0–255)",
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            anchor="w",
            padx=15
        )

        self.analysis_chart_frame = tk.Frame(
            histogram_card,
            bg=self.card_color
        )

        self.analysis_chart_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        bottom = tk.Frame(
            self.analysis_window,
            bg=self.bg_color
        )

        bottom.pack(
            fill="x",
            padx=15,
            pady=(0, 15)
        )

        ttk.Button(
            bottom,
            text="Save Histogram",
            command=self.save_histogram
        ).pack(
            side="right"
        )

        self.update_analysis_window()

    def update_analysis_window(self):

        if self.current_image is None:
            return

        if self.analysis_window is None:
            return

        try:

            if not self.analysis_window.winfo_exists():
                return

        except tk.TclError:
            return

        image = self.current_image

        image_array = np.array(
            image
        )

        if image_array.ndim == 2:

            gray_array = image_array
            rgb_array = None

        else:

            rgb_array = image_array[:, :, :3]
            gray_array = None

        for widget in (
            self.analysis_stats_frame
            .winfo_children()
        ):

            widget.destroy()

        tk.Label(
            self.analysis_stats_frame,
            text="Image Statistics",
            font=("Segoe UI", 13, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(15, 10)
        )

        self.create_analysis_section(
            self.analysis_stats_frame,
            "General Information"
        )

        image_format = (
            image.format
            if image.format
            else (
                self.image_path.split(".")[-1].upper()
                if self.image_path
                else "Processed"
            )
        )

        rows = [
            (
                "Dimensions",
                f"{image.width} × {image.height} px"
            ),
            (
                "Width",
                f"{image.width} px"
            ),
            (
                "Height",
                f"{image.height} px"
            ),
            (
                "Color Mode",
                image.mode
            ),
            (
                "Format",
                str(image_format)
            ),
            (
                "Total Pixels",
                f"{image.width * image.height:,}"
            )
        ]

        for label, value in rows:

            self.create_stat_row(
                self.analysis_stats_frame,
                label,
                value
            )

        self.create_analysis_section(
            self.analysis_stats_frame,
            "Overall Pixel Statistics"
        )

        self.create_stat_row(
            self.analysis_stats_frame,
            "Minimum",
            str(int(image_array.min()))
        )

        self.create_stat_row(
            self.analysis_stats_frame,
            "Maximum",
            str(int(image_array.max()))
        )

        self.create_stat_row(
            self.analysis_stats_frame,
            "Mean",
            f"{float(image_array.mean()):.2f}"
        )

        self.create_analysis_section(
            self.analysis_stats_frame,
            "RGB Channel Statistics"
        )

        if rgb_array is not None:

            channels = [
                ("Red", 0),
                ("Green", 1),
                ("Blue", 2)
            ]

            for name, index in channels:

                channel = rgb_array[:, :, index]

                value = (
                    f"Min {int(channel.min())} | "
                    f"Max {int(channel.max())} | "
                    f"Mean {float(channel.mean()):.2f}"
                )

                self.create_stat_row(
                    self.analysis_stats_frame,
                    name,
                    value
                )

        else:

            self.create_stat_row(
                self.analysis_stats_frame,
                "Grayscale",
                "Single channel image"
            )

        self.create_analysis_section(
            self.analysis_stats_frame,
            "Current Processing"
        )

        rows = [
            (
                "Brightness",
                f"{self.brightness_value:+d}"
            ),
            (
                "Contrast",
                f"{self.contrast_value:+d}"
            ),
            (
                "Filter",
                self.current_filter
            ),
            (
                "Advanced Mode",
                self.advanced_mode
            ),
            (
                "Saturation",
                f"{self.saturation_value:+d}"
            ),
            (
                "Hue",
                f"{self.hue_value:+d}°"
            ),
            (
                "RGB Adjustments",
                (
                    f"R {self.red_value:+d} | "
                    f"G {self.green_value:+d} | "
                    f"B {self.blue_value:+d}"
                )
            ),
            (
                "Gamma",
                f"{self.gamma_value:.1f}"
            ),
            (
                "Threshold",
                str(self.threshold_value)
            ),
            (
                "Zoom",
                (
                    "Fit"
                    if self.zoom_mode == "fit"
                    else f"{self.zoom_level * 100:.0f}%"
                )
            )
        ]

        for label, value in rows:

            self.create_stat_row(
                self.analysis_stats_frame,
                label,
                value
            )

        self.create_histogram(
            rgb_array,
            gray_array
        )

    def create_analysis_section(
        self,
        parent,
        text
    ):

        tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=self.card_color,
            fg=self.accent_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(10, 4)
        )

    def create_stat_row(
        self,
        parent,
        label,
        value
    ):

        row = tk.Frame(
            parent,
            bg=self.card_color
        )

        row.pack(
            fill="x",
            padx=15,
            pady=2
        )

        tk.Label(
            row,
            text=label,
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            side="left"
        )

        tk.Label(
            row,
            text=value,
            font=("Segoe UI", 9, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            side="right"
        )

    # ============================================================
    # HISTOGRAM
    # ============================================================

    def create_histogram(
        self,
        rgb_array,
        gray_array
    ):

        if not hasattr(
            self,
            "analysis_chart_frame"
        ):
            return

        for widget in (
            self.analysis_chart_frame
            .winfo_children()
        ):

            widget.destroy()

        self.analysis_figure = Figure(
            figsize=(7, 5),
            dpi=100
        )

        axis = self.analysis_figure.add_subplot(
            111
        )

        if rgb_array is not None:

            red = rgb_array[:, :, 0].ravel()
            green = rgb_array[:, :, 1].ravel()
            blue = rgb_array[:, :, 2].ravel()

            red_hist = np.bincount(
                red,
                minlength=256
            )

            green_hist = np.bincount(
                green,
                minlength=256
            )

            blue_hist = np.bincount(
                blue,
                minlength=256
            )

            x = np.arange(256)

            axis.plot(
                x,
                red_hist,
                color="red",
                linewidth=1.2,
                label="Red"
            )

            axis.plot(
                x,
                green_hist,
                color="green",
                linewidth=1.2,
                label="Green"
            )

            axis.plot(
                x,
                blue_hist,
                color="blue",
                linewidth=1.2,
                label="Blue"
            )

        else:

            histogram = np.bincount(
                gray_array.ravel(),
                minlength=256
            )

            x = np.arange(256)

            axis.plot(
                x,
                histogram,
                color="gray",
                linewidth=1.5,
                label="Grayscale"
            )

        axis.set_title(
            "Pixel Intensity Distribution"
        )

        axis.set_xlabel(
            "Pixel Intensity"
        )

        axis.set_ylabel(
            "Pixel Count"
        )

        axis.set_xlim(
            0,
            255
        )

        axis.grid(
            True,
            alpha=0.2
        )

        axis.legend(
            loc="upper right"
        )

        self.analysis_figure.tight_layout()

        self.analysis_canvas = FigureCanvasTkAgg(
            self.analysis_figure,
            master=self.analysis_chart_frame
        )

        self.analysis_canvas.draw()

        self.analysis_canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def save_histogram(self):

        if self.analysis_figure is None:

            messagebox.showwarning(
                "No Histogram",
                "There is no histogram available."
            )

            return

        file_path = filedialog.asksaveasfilename(
            title="Save Histogram",
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PDF Files", "*.pdf"),
                ("SVG Files", "*.svg")
            ]
        )

        if not file_path:
            return

        try:

            self.analysis_figure.savefig(
                file_path,
                dpi=200,
                bbox_inches="tight"
            )

            messagebox.showinfo(
                "Histogram Saved",
                f"Histogram successfully saved to:\n\n{file_path}"
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                f"Unable to save histogram.\n\n{error}"
            )

    def close_analysis_window(self):

        try:

            self.analysis_window.destroy()

        except (
            tk.TclError,
            AttributeError
        ):
            pass

        self.analysis_window = None
        self.analysis_figure = None
        self.analysis_canvas = None

    # ============================================================
    # SECONDARY WINDOWS
    # ============================================================

    def refresh_secondary_windows(self):

        try:

            if (
                self.analysis_window is not None
                and self.analysis_window.winfo_exists()
            ):

                self.update_analysis_window()

        except tk.TclError:
            pass

        try:

            if (
                self.comparison_window is not None
                and self.comparison_window.winfo_exists()
            ):

                self.update_comparison_window()

        except tk.TclError:
            pass

    # ============================================================
    # CLOSE ACTIVE WINDOW
    # ============================================================

    def close_active_window(self):

        windows = [
            self.analysis_window,
            self.comparison_window,
            self.history_window
        ]

        for window in windows:

            if window is not None:

                try:

                    if window.winfo_exists():

                        window.destroy()

                except tk.TclError:
                    pass

        self.analysis_window = None
        self.comparison_window = None
        self.history_window = None
        self.history_listbox = None

    # ============================================================
    # IMAGE INFORMATION / METADATA
    # ============================================================

    def show_metadata(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        if self.metadata_window is not None:

            try:

                if self.metadata_window.winfo_exists():

                    self.update_metadata_window()
                    self.metadata_window.lift()
                    return

            except tk.TclError:
                pass

        self.metadata_window = tk.Toplevel(
            self.root
        )

        self.metadata_window.title(
            "ImageLab - Image Information & Metadata"
        )

        self.metadata_window.geometry(
            "1000x680"
        )

        self.metadata_window.minsize(
            850,
            550
        )

        self.metadata_window.configure(
            bg=self.bg_color
        )

        self.metadata_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_metadata_window
        )

        header = tk.Frame(
            self.metadata_window,
            bg=self.card_color,
            height=70
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        tk.Label(
            header,
            text="Image Information & Metadata",
            font=("Segoe UI", 20, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            side="left",
            padx=20
        )

        ttk.Button(
            header,
            text="Refresh",
            command=self.update_metadata_window
        ).pack(
            side="right",
            padx=10
        )

        ttk.Button(
            header,
            text="Copy Information",
            command=self.copy_metadata_information
        ).pack(
            side="right",
            padx=10
        )

        content = tk.Frame(
            self.metadata_window,
            bg=self.bg_color
        )

        content.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        content.grid_columnconfigure(
            0,
            weight=1
        )

        content.grid_columnconfigure(
            1,
            weight=1
        )

        content.grid_rowconfigure(
            0,
            weight=1
        )

        left = tk.Frame(
            content,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 7)
        )

        right = tk.Frame(
            content,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(7, 0)
        )

        tk.Label(
            left,
            text="File & Image Information",
            font=("Segoe UI", 13, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(15, 8)
        )

        left_frame = tk.Frame(
            left,
            bg=self.card_color
        )

        left_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )

        left_canvas = tk.Canvas(
            left_frame,
            bg=self.card_color,
            highlightthickness=0
        )

        left_scroll = ttk.Scrollbar(
            left_frame,
            orient="vertical",
            command=left_canvas.yview
        )

        self.metadata_left_inner = tk.Frame(
            left_canvas,
            bg=self.card_color
        )

        self.metadata_left_window = left_canvas.create_window(
            (0, 0),
            window=self.metadata_left_inner,
            anchor="nw"
        )

        self.metadata_left_inner.bind(
            "<Configure>",
            lambda event: left_canvas.configure(
                scrollregion=left_canvas.bbox("all")
            )
        )

        left_canvas.bind(
            "<Configure>",
            lambda event: left_canvas.itemconfigure(
                self.metadata_left_window,
                width=event.width
            )
        )

        left_canvas.configure(
            yscrollcommand=left_scroll.set
        )

        left_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        left_scroll.pack(
            side="right",
            fill="y"
        )

        tk.Label(
            right,
            text="EXIF / Camera Metadata",
            font=("Segoe UI", 13, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(15, 8)
        )

        right_frame = tk.Frame(
            right,
            bg=self.card_color
        )

        right_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )

        right_canvas = tk.Canvas(
            right_frame,
            bg=self.card_color,
            highlightthickness=0
        )

        right_scroll = ttk.Scrollbar(
            right_frame,
            orient="vertical",
            command=right_canvas.yview
        )

        self.metadata_right_inner = tk.Frame(
            right_canvas,
            bg=self.card_color
        )

        self.metadata_right_window = right_canvas.create_window(
            (0, 0),
            window=self.metadata_right_inner,
            anchor="nw"
        )

        self.metadata_right_inner.bind(
            "<Configure>",
            lambda event: right_canvas.configure(
                scrollregion=right_canvas.bbox("all")
            )
        )

        right_canvas.bind(
            "<Configure>",
            lambda event: right_canvas.itemconfigure(
                self.metadata_right_window,
                width=event.width
            )
        )

        right_canvas.configure(
            yscrollcommand=right_scroll.set
        )

        right_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        right_scroll.pack(
            side="right",
            fill="y"
        )

        self.update_metadata_window()

    def _metadata_section(self, parent, title):

        tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 10, "bold"),
            bg=self.card_color,
            fg=self.accent_color
        ).pack(
            anchor="w",
            fill="x",
            pady=(10, 5)
        )

    def _metadata_row(self, parent, label, value):

        frame = tk.Frame(
            parent,
            bg=self.card_color
        )

        frame.pack(
            fill="x",
            pady=2
        )

        tk.Label(
            frame,
            text=label,
            font=("Segoe UI", 9, "bold"),
            bg=self.card_color,
            fg=self.text_color,
            width=19,
            anchor="w"
        ).pack(
            side="left",
            anchor="w"
        )

        tk.Label(
            frame,
            text=str(value),
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text,
            anchor="w",
            justify="left",
            wraplength=330
        ).pack(
            side="left",
            fill="x",
            expand=True
        )

    def _metadata_file_size(self):

        if not self.image_path:
            return "Unavailable"

        try:

            size = Path(
                self.image_path
            ).stat().st_size

        except OSError:
            return "Unavailable"

        units = [
            "B",
            "KB",
            "MB",
            "GB"
        ]

        value = float(size)
        unit = units[0]

        for unit in units:

            if value < 1024 or unit == units[-1]:
                break

            value /= 1024

        return f"{value:.2f} {unit}"

    def _metadata_exif_rows(self):

        if not self.image_path:
            return []

        try:

            with Image.open(
                self.image_path
            ) as source:

                exif = source.getexif()

                if not exif:
                    return []

                rows = []

                for tag_id, value in exif.items():

                    tag_name = ExifTags.TAGS.get(
                        tag_id,
                        str(tag_id)
                    )

                    if isinstance(value, bytes):

                        try:
                            value = value.decode(
                                "utf-8",
                                errors="replace"
                            )
                        except Exception:
                            value = repr(value)

                    value_text = str(value)

                    if len(value_text) > 250:

                        value_text = (
                            value_text[:247]
                            + "..."
                        )

                    rows.append(
                        (
                            str(tag_name),
                            value_text
                        )
                    )

                rows.sort(
                    key=lambda item: item[0].lower()
                )

                return rows

        except Exception:
            return []

    def _metadata_current_processing_rows(self):

        return [
            (
                "Brightness",
                f"{self.brightness_value:+d}"
            ),
            (
                "Contrast",
                f"{self.contrast_value:+d}"
            ),
            (
                "Saturation",
                f"{self.saturation_value:+d}"
            ),
            (
                "Hue",
                f"{self.hue_value:+d}°"
            ),
            (
                "RGB Adjustment",
                (
                    f"R {self.red_value:+d} | "
                    f"G {self.green_value:+d} | "
                    f"B {self.blue_value:+d}"
                )
            ),
            (
                "Gamma",
                f"{self.gamma_value:.1f}"
            ),
            (
                "Threshold",
                str(self.threshold_value)
            ),
            (
                "Advanced Mode",
                self.advanced_mode
            ),
            (
                "Filter",
                self.current_filter
            ),
            (
                "Zoom",
                (
                    "Fit"
                    if self.zoom_mode == "fit"
                    else f"{self.zoom_level * 100:.0f}%"
                )
            )
        ]

    def update_metadata_window(self):

        if self.current_image is None:
            return

        if self.metadata_window is None:
            return

        try:

            if not self.metadata_window.winfo_exists():
                return

        except tk.TclError:
            return

        for widget in (
            self.metadata_left_inner,
            self.metadata_right_inner
        ):

            for child in widget.winfo_children():
                child.destroy()

        image = self.current_image
        image_array = np.array(image)

        # --------------------------------------------------------
        # File / image information
        # --------------------------------------------------------

        self._metadata_section(
            self.metadata_left_inner,
            "File Information"
        )

        filename = (
            Path(self.image_path).name
            if self.image_path
            else "Processed image"
        )

        image_format = (
            image.format
            if image.format
            else (
                Path(self.image_path).suffix.upper().lstrip(".")
                if self.image_path
                else "Processed"
            )
        )

        file_rows = [
            ("Filename", filename),
            ("File Type", str(image_format)),
            ("File Size", self._metadata_file_size()),
            ("File Path", self.image_path or "Not saved / processed image"),
        ]

        for label, value in file_rows:
            self._metadata_row(
                self.metadata_left_inner,
                label,
                value
            )

        self._metadata_section(
            self.metadata_left_inner,
            "Image Properties"
        )

        dpi = None

        if self.image_path:

            try:

                with Image.open(
                    self.image_path
                ) as source:

                    dpi = source.info.get("dpi")

            except Exception:
                dpi = None

        dpi_text = (
            f"{dpi[0]:.0f} × {dpi[1]:.0f} DPI"
            if isinstance(dpi, tuple)
            and len(dpi) >= 2
            else "Not specified"
        )

        properties = [
            (
                "Dimensions",
                f"{image.width} × {image.height} px"
            ),
            (
                "Width",
                f"{image.width:,} px"
            ),
            (
                "Height",
                f"{image.height:,} px"
            ),
            (
                "Total Pixels",
                f"{image.width * image.height:,}"
            ),
            (
                "Color Mode",
                image.mode
            ),
            (
                "Channels",
                str(
                    len(image.getbands())
                )
            ),
            (
                "Resolution",
                dpi_text
            ),
        ]

        for label, value in properties:
            self._metadata_row(
                self.metadata_left_inner,
                label,
                value
            )

        self._metadata_section(
            self.metadata_left_inner,
            "Pixel Statistics"
        )

        stats = [
            (
                "Minimum",
                int(image_array.min())
            ),
            (
                "Maximum",
                int(image_array.max())
            ),
            (
                "Mean",
                f"{float(image_array.mean()):.2f}"
            )
        ]

        for label, value in stats:
            self._metadata_row(
                self.metadata_left_inner,
                label,
                value
            )

        if image_array.ndim >= 3:

            self._metadata_section(
                self.metadata_left_inner,
                "RGB Channel Statistics"
            )

            channels = [
                ("Red", 0),
                ("Green", 1),
                ("Blue", 2)
            ]

            for name, index in channels:

                channel = image_array[:, :, index]

                self._metadata_row(
                    self.metadata_left_inner,
                    name,
                    (
                        f"Min {int(channel.min())} | "
                        f"Max {int(channel.max())} | "
                        f"Mean {float(channel.mean()):.2f}"
                    )
                )

        # --------------------------------------------------------
        # Processing information
        # --------------------------------------------------------

        self._metadata_section(
            self.metadata_left_inner,
            "Current Processing"
        )

        for label, value in self._metadata_current_processing_rows():
            self._metadata_row(
                self.metadata_left_inner,
                label,
                value
            )

        # --------------------------------------------------------
        # EXIF
        # --------------------------------------------------------

        exif_rows = self._metadata_exif_rows()

        if exif_rows:

            for label, value in exif_rows:
                self._metadata_row(
                    self.metadata_right_inner,
                    label,
                    value
                )

        else:

            tk.Label(
                self.metadata_right_inner,
                text=(
                    "No EXIF metadata is available for this image.\n\n"
                    "EXIF information is normally stored by cameras "
                    "and mobile devices in supported image files."
                ),
                font=("Segoe UI", 10),
                bg=self.card_color,
                fg=self.secondary_text,
                justify="left",
                wraplength=400
            ).pack(
                anchor="w",
                fill="x",
                pady=(10, 20)
            )

        self._metadata_section(
            self.metadata_right_inner,
            "Application Information"
        )

        application_rows = [
            (
                "ImageLab Version",
                "Step 18"
            ),
            (
                "Undo Steps",
                str(len(self.undo_stack))
            ),
            (
                "Redo Steps",
                str(len(self.redo_stack))
            ),
            (
                "History Entries",
                str(len(self.history))
            )
        ]

        for label, value in application_rows:
            self._metadata_row(
                self.metadata_right_inner,
                label,
                value
            )

    def _metadata_information_text(self):

        if self.current_image is None:
            return "No image loaded."

        image = self.current_image
        image_array = np.array(image)

        lines = [
            "ImageLab - Image Information",
            "=" * 36,
            "",
            f"Filename: {Path(self.image_path).name if self.image_path else 'Processed image'}",
            f"Path: {self.image_path or 'Not available'}",
            f"Format: {image.format or 'Processed'}",
            f"File Size: {self._metadata_file_size()}",
            f"Dimensions: {image.width} x {image.height} px",
            f"Pixels: {image.width * image.height:,}",
            f"Color Mode: {image.mode}",
            f"Channels: {len(image.getbands())}",
            "",
            "Pixel Statistics",
            f"Minimum: {int(image_array.min())}",
            f"Maximum: {int(image_array.max())}",
            f"Mean: {float(image_array.mean()):.2f}",
            "",
            "Current Processing"
        ]

        for label, value in self._metadata_current_processing_rows():
            lines.append(
                f"{label}: {value}"
            )

        exif_rows = self._metadata_exif_rows()

        lines.extend([
            "",
            "EXIF / Camera Metadata"
        ])

        if exif_rows:

            for label, value in exif_rows:
                lines.append(
                    f"{label}: {value}"
                )

        else:
            lines.append(
                "No EXIF metadata available."
            )

        return "\n".join(lines)

    def copy_metadata_information(self):

        information = self._metadata_information_text()

        try:

            self.root.clipboard_clear()
            self.root.clipboard_append(
                information
            )
            self.root.update()

            messagebox.showinfo(
                "Copied",
                "Image information has been copied to the clipboard."
            )

        except tk.TclError as error:

            messagebox.showerror(
                "Clipboard Error",
                f"Could not copy the information.\n\n{error}"
            )

    def close_metadata_window(self):

        try:
            self.metadata_window.destroy()
        except (
            tk.TclError,
            AttributeError
        ):
            pass

        self.metadata_window = None
        self.metadata_content = None

    # ============================================================
    # BATCH PROCESSING
    # ============================================================

    def show_batch_processing(self):

        if self.batch_window is not None:

            try:

                if self.batch_window.winfo_exists():

                    self.batch_window.lift()
                    return

            except tk.TclError:
                pass

        self.batch_window = tk.Toplevel(self.root)
        self.batch_window.title("ImageLab - Batch Processing")
        self.batch_window.geometry("1120x720")
        self.batch_window.minsize(950, 620)
        self.batch_window.configure(bg=self.bg_color)
        self.batch_window.transient(self.root)
        self.batch_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_batch_processing
        )

        header = tk.Frame(
            self.batch_window,
            bg=self.card_color,
            height=65
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Batch Image Processing",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(side="left", padx=20)

        tk.Label(
            header,
            text="Apply the same ImageLab processing settings to multiple images",
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(side="left", padx=5)

        content = tk.Frame(
            self.batch_window,
            bg=self.bg_color
        )
        content.pack(fill="both", expand=True, padx=15, pady=12)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=0)

        # Queue card
        queue_card = tk.Frame(
            content,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )
        queue_card.grid(
            row=0, column=0, rowspan=2,
            sticky="nsew", padx=(0, 7)
        )

        tk.Label(
            queue_card,
            text="Image Queue",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(anchor="w", padx=15, pady=(12, 5))

        qbuttons = tk.Frame(queue_card, bg=self.card_color)
        qbuttons.pack(fill="x", padx=12, pady=5)

        ttk.Button(
            qbuttons, text="Add Images", command=self.batch_add_images
        ).pack(side="left", padx=3)
        ttk.Button(
            qbuttons, text="Add Folder", command=self.batch_add_folder
        ).pack(side="left", padx=3)
        ttk.Button(
            qbuttons, text="Clear", command=self.batch_clear_files
        ).pack(side="left", padx=3)

        list_frame = tk.Frame(queue_card, bg=self.card_color)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(5, 12))

        self.batch_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 9),
            bg="#f9fafb",
            fg=self.text_color,
            selectbackground=self.accent_color,
            selectforeground="white",
            activestyle="none"
        )
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.batch_listbox.yview
        )
        self.batch_listbox.configure(yscrollcommand=scrollbar.set)
        self.batch_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Settings card
        settings_card = tk.Frame(
            content,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )
        settings_card.grid(
            row=0, column=1, sticky="nsew", padx=(7, 0)
        )

        tk.Label(
            settings_card,
            text="Processing Settings",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(anchor="w", padx=15, pady=(12, 8))

        settings = tk.Frame(settings_card, bg=self.card_color)
        settings.pack(fill="both", expand=True, padx=15, pady=(0, 8))
        settings.grid_columnconfigure(1, weight=1)

        self.batch_brightness_var = tk.IntVar(value=self.brightness_value)
        self.batch_contrast_var = tk.IntVar(value=self.contrast_value)
        self.batch_saturation_var = tk.IntVar(value=self.saturation_value)
        self.batch_hue_var = tk.IntVar(value=self.hue_value)
        self.batch_red_var = tk.IntVar(value=self.red_value)
        self.batch_green_var = tk.IntVar(value=self.green_value)
        self.batch_blue_var = tk.IntVar(value=self.blue_value)
        self.batch_gamma_var = tk.DoubleVar(value=self.gamma_value)
        self.batch_threshold_var = tk.IntVar(value=self.threshold_value)

        self._batch_slider(settings, 0, "Brightness", self.batch_brightness_var, -100, 100)
        self._batch_slider(settings, 1, "Contrast", self.batch_contrast_var, -100, 100)
        self._batch_slider(settings, 2, "Saturation", self.batch_saturation_var, -100, 100)
        self._batch_slider(settings, 3, "Hue", self.batch_hue_var, -180, 180)
        self._batch_slider(settings, 4, "Red", self.batch_red_var, -100, 100)
        self._batch_slider(settings, 5, "Green", self.batch_green_var, -100, 100)
        self._batch_slider(settings, 6, "Blue", self.batch_blue_var, -100, 100)
        self._batch_slider(settings, 7, "Gamma", self.batch_gamma_var, 0.1, 3.0, 0.1)
        self._batch_slider(settings, 8, "Threshold", self.batch_threshold_var, 0, 255)

        tk.Label(
            settings, text="Filter", font=("Segoe UI", 9, "bold"),
            bg=self.card_color, fg=self.text_color, anchor="w"
        ).grid(row=9, column=0, sticky="w", pady=4)

        self.batch_filter_var = tk.StringVar(value=self.current_filter)
        ttk.Combobox(
            settings,
            textvariable=self.batch_filter_var,
            values=[
                "Original", "Grayscale", "Sepia", "Negative", "Blur",
                "Sharpen", "Edge Enhance", "Emboss", "Smooth", "Contour"
            ],
            state="readonly"
        ).grid(row=9, column=1, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(
            settings, text="Advanced", font=("Segoe UI", 9, "bold"),
            bg=self.card_color, fg=self.text_color, anchor="w"
        ).grid(row=10, column=0, sticky="w", pady=4)

        self.batch_advanced_var = tk.StringVar(value=self.advanced_mode)
        ttk.Combobox(
            settings,
            textvariable=self.batch_advanced_var,
            values=[
                "None", "Histogram Equalization", "Auto Enhance", "Threshold"
            ],
            state="readonly"
        ).grid(row=10, column=1, sticky="ew", padx=(8, 0), pady=4)

        # Output card
        output_card = tk.Frame(
            content, bg=self.card_color, bd=1, relief="solid"
        )
        output_card.grid(
            row=1, column=1, sticky="ew", padx=(7, 0), pady=(10, 0)
        )

        tk.Label(
            output_card, text="Output", font=("Segoe UI", 12, "bold"),
            bg=self.card_color, fg=self.text_color
        ).pack(anchor="w", padx=15, pady=(10, 5))

        out = tk.Frame(output_card, bg=self.card_color)
        out.pack(fill="x", padx=15, pady=(0, 12))
        out.grid_columnconfigure(1, weight=1)

        tk.Label(
            out, text="Folder", font=("Segoe UI", 9, "bold"),
            bg=self.card_color, fg=self.text_color
        ).grid(row=0, column=0, sticky="w")

        self.batch_output_var = tk.StringVar(value="")
        ttk.Entry(out, textvariable=self.batch_output_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            out, text="Browse", command=self.batch_choose_output
        ).grid(row=0, column=2)

        tk.Label(
            out, text="Format", font=("Segoe UI", 9, "bold"),
            bg=self.card_color, fg=self.text_color
        ).grid(row=1, column=0, sticky="w", pady=(7, 0))

        self.batch_format_var = tk.StringVar(value="PNG")
        ttk.Combobox(
            out, textvariable=self.batch_format_var,
            values=["PNG", "JPEG", "BMP", "TIFF"],
            state="readonly", width=10
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(7, 0))

        tk.Label(
            out, text="Resize", font=("Segoe UI", 9, "bold"),
            bg=self.card_color, fg=self.text_color
        ).grid(row=1, column=2, sticky="w", pady=(7, 0))

        self.batch_resize_var = tk.StringVar(value="Original Size")
        ttk.Combobox(
            out, textvariable=self.batch_resize_var,
            values=[
                "Original Size", "25%", "50%", "75%", "125%", "150%", "200%"
            ],
            state="readonly", width=14
        ).grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(7, 0))

        self.batch_progress = ttk.Progressbar(
            out, mode="determinate", maximum=100
        )
        self.batch_progress.grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(10, 4)
        )

        self.batch_status_label = tk.Label(
            out, text="Ready - add images or a folder",
            font=("Segoe UI", 9), bg=self.card_color,
            fg=self.secondary_text, anchor="w"
        )
        self.batch_status_label.grid(
            row=3, column=0, columnspan=3, sticky="ew"
        )

        self.batch_process_button = ttk.Button(
            out, text="Process All Images", command=self.batch_process_images
        )
        self.batch_process_button.grid(
            row=3, column=3, sticky="e", padx=(8, 0)
        )

    def _batch_slider(self, parent, row, label, variable, minimum, maximum, resolution=1):

        tk.Label(
            parent, text=label, font=("Segoe UI", 9, "bold"),
            bg=self.card_color, fg=self.text_color, anchor="w"
        ).grid(row=row, column=0, sticky="w", pady=2)

        scale = tk.Scale(
            parent, variable=variable, from_=minimum, to=maximum,
            resolution=resolution, orient="horizontal", showvalue=False,
            bg=self.card_color, highlightthickness=0, troughcolor="#d1d5db"
        )
        scale.grid(row=row, column=1, sticky="ew", padx=8, pady=2)

        tk.Label(
            parent, textvariable=variable, width=6,
            font=("Segoe UI", 8), bg=self.card_color,
            fg=self.secondary_text
        ).grid(row=row, column=2, sticky="e")

    def batch_add_images(self):

        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif")
            ]
        )

        for file_path in files:

            if file_path not in self.batch_files:

                self.batch_files.append(file_path)
                self.batch_listbox.insert(tk.END, file_path)

        self.batch_update_status(
            f"{len(self.batch_files)} image(s) in queue"
        )

    def batch_add_folder(self):

        folder = filedialog.askdirectory(title="Select Image Folder")

        if not folder:
            return

        supported = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}

        try:
            files = sorted(
                [
                    str(item) for item in Path(folder).iterdir()
                    if item.is_file() and item.suffix.lower() in supported
                ],
                key=lambda value: Path(value).name.lower()
            )
        except OSError as error:
            messagebox.showerror(
                "Folder Error",
                f"Could not read the selected folder.\n\n{error}"
            )
            return

        for file_path in files:

            if file_path not in self.batch_files:

                self.batch_files.append(file_path)
                self.batch_listbox.insert(tk.END, file_path)

        self.batch_update_status(
            f"{len(self.batch_files)} image(s) in queue"
        )

    def batch_clear_files(self):

        self.batch_files.clear()
        self.batch_listbox.delete(0, tk.END)
        self.batch_progress["value"] = 0
        self.batch_update_status("Queue cleared - add images or a folder")

    def batch_choose_output(self):

        folder = filedialog.askdirectory(title="Select Output Folder")

        if folder:
            self.batch_output_var.set(folder)
            self.batch_update_status(f"Output folder: {folder}")

    def batch_update_status(self, text):

        if self.batch_status_label is not None:

            try:
                self.batch_status_label.config(text=text)
            except tk.TclError:
                pass

    def _batch_resize_image(self, image, option):

        scales = {
            "25%": 0.25,
            "50%": 0.50,
            "75%": 0.75,
            "125%": 1.25,
            "150%": 1.50,
            "200%": 2.00
        }

        scale = scales.get(option)

        if scale is None:
            return image

        size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale))
        )

        return image.resize(
            size,
            Image.Resampling.LANCZOS
        )

    def _batch_process_one(self, source_path, output_path):

        with Image.open(source_path) as source:
            source_image = source.copy()

        # Preserve all existing ImageLab processing behavior by using
        # the same processing pipeline as the main editor.
        saved_state = {
            "original_image": self.original_image,
            "base_image": self.base_image,
            "current_image": self.current_image,
            "brightness_value": self.brightness_value,
            "contrast_value": self.contrast_value,
            "saturation_value": self.saturation_value,
            "hue_value": self.hue_value,
            "red_value": self.red_value,
            "green_value": self.green_value,
            "blue_value": self.blue_value,
            "gamma_value": self.gamma_value,
            "threshold_value": self.threshold_value,
            "advanced_mode": self.advanced_mode,
            "current_filter": self.current_filter,
            "image_path": self.image_path,
        }

        try:

            self.base_image = source_image
            self.current_image = source_image.copy()
            self.brightness_value = int(self.batch_brightness_var.get())
            self.contrast_value = int(self.batch_contrast_var.get())
            self.saturation_value = int(self.batch_saturation_var.get())
            self.hue_value = int(self.batch_hue_var.get())
            self.red_value = int(self.batch_red_var.get())
            self.green_value = int(self.batch_green_var.get())
            self.blue_value = int(self.batch_blue_var.get())
            self.gamma_value = float(self.batch_gamma_var.get())
            self.threshold_value = int(self.batch_threshold_var.get())
            self.advanced_mode = self.batch_advanced_var.get()
            self.current_filter = self.batch_filter_var.get()

            # Batch processing always uses full resolution.
            self.apply_adjustments(
                preview=False,
                refresh_secondary=False
            )
            result = self.current_image.copy()

            result = self._batch_resize_image(
                result,
                self.batch_resize_var.get()
            )

            output_format = self.batch_format_var.get()

            if output_format == "JPEG" and result.mode not in ("RGB", "L"):
                result = result.convert("RGB")

            extension = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "BMP": ".bmp",
                "TIFF": ".tiff"
            }[output_format]

            stem = Path(source_path).stem
            destination = Path(output_path) / f"{stem}_processed{extension}"
            counter = 1

            while destination.exists():

                destination = (
                    Path(output_path)
                    / f"{stem}_processed_{counter}{extension}"
                )
                counter += 1

            save_kwargs = {}

            if output_format == "JPEG":
                save_kwargs = {"quality": 95, "optimize": True}

            result.save(
                destination,
                format=output_format,
                **save_kwargs
            )

            return destination

        finally:

            for key, value in saved_state.items():
                setattr(self, key, value)

    def batch_process_images(self):

        if not self.batch_files:

            messagebox.showwarning(
                "No Images",
                "Please add images or a folder to the queue first."
            )
            return

        output_folder = self.batch_output_var.get().strip()

        if not output_folder:

            output_folder = filedialog.askdirectory(
                title="Select Output Folder"
            )

            if not output_folder:
                return

            self.batch_output_var.set(output_folder)

        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            messagebox.showerror(
                "Output Error",
                f"Could not create/access the output folder.\n\n{error}"
            )
            return

        total = len(self.batch_files)
        successful = 0
        failed = []

        self.batch_progress["value"] = 0
        self.batch_process_button.config(
            state="disabled",
            text="Processing..."
        )

        try:

            for index, file_path in enumerate(self.batch_files, start=1):

                self.batch_update_status(
                    f"Processing {index} of {total}: {Path(file_path).name}"
                )

                try:
                    self._batch_process_one(
                        file_path,
                        output_folder
                    )
                    successful += 1

                except Exception as error:
                    failed.append(
                        f"{Path(file_path).name}: {error}"
                    )

                self.batch_progress["value"] = (index / total) * 100
                self.batch_window.update_idletasks()

        finally:

            self.batch_process_button.config(
                state="normal",
                text="Process All Images"
            )

            # Restore the editor preview exactly as it was before batch work.
            if self.base_image is not None:
                self.apply_adjustments()

        self.batch_update_status(
            f"Finished • {successful} succeeded • {len(failed)} failed"
        )

        if not failed:

            messagebox.showinfo(
                "Batch Processing Complete",
                f"Successfully processed {successful} image(s).\n\n"
                f"Output folder:\n{output_folder}"
            )

        else:

            details = "\n".join(failed[:10])

            if len(failed) > 10:
                details += f"\n... and {len(failed) - 10} more."

            messagebox.showwarning(
                "Batch Processing Finished",
                f"Successful: {successful}\n"
                f"Failed: {len(failed)}\n\n"
                f"Output folder:\n{output_folder}\n\n"
                f"Errors:\n{details}"
            )

    def close_batch_processing(self):

        if self.batch_window is not None:

            try:
                self.batch_window.destroy()
            except tk.TclError:
                pass

        self.batch_window = None
        self.batch_listbox = None
        self.batch_progress = None
        self.batch_status_label = None
        self.batch_process_button = None

    # ============================================================
    # PRESETS
    # ============================================================

    def show_presets(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        window = tk.Toplevel(self.root)
        window.title("ImageLab - Editing Presets")
        window.geometry("860x760")
        window.minsize(760, 700)
        window.configure(bg=self.bg_color)
        window.transient(self.root)
        window.lift()
        window.focus_force()

        # --------------------------------------------------------
        # HEADER
        # --------------------------------------------------------

        header = tk.Frame(
            window,
            bg=self.card_color,
            height=82
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Editing Presets",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=22,
            pady=(12, 0)
        )

        tk.Label(
            header,
            text="Apply a complete combination of ImageLab adjustments with one click",
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            anchor="w",
            padx=22
        )

        # --------------------------------------------------------
        # PRESET GRID
        # --------------------------------------------------------

        content = tk.Frame(
            window,
            bg=self.bg_color
        )
        content.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=10
        )

        for column in range(3):
            content.grid_columnconfigure(column, weight=1, uniform="preset_col")

        for row in range(5):
            content.grid_rowconfigure(row, weight=1, minsize=105)

        presets = [
            ("Auto Enhance", "Automatic tonal correction", "Auto Enhance"),
            ("Brighten", "Lift overall brightness", "Brighten"),
            ("Darken", "Reduce brightness for mood", "Darken"),
            ("Vivid", "More contrast and color", "Vivid"),
            ("Color Boost", "Increase saturation", "Color Boost"),
            ("Black & White", "Convert to monochrome", "Black & White"),
            ("Portrait", "Soft contrast and natural color", "Portrait"),
            ("Warm", "Add a warmer color tone", "Warm"),
            ("Cool", "Add a cooler color tone", "Cool"),
            ("Sharp", "Improve edge definition", "Sharp"),
            ("Soft", "Create a softer appearance", "Soft"),
            ("Vintage", "Warm muted photographic style", "Vintage"),
            ("Cinematic", "Contrast with restrained color", "Cinematic"),
            ("Reset to Original", "Clear editing adjustments", "Reset to Original")
        ]

        for index, (name, description, preset_key) in enumerate(presets):

            row = index // 3
            column = index % 3

            card = tk.Frame(
                content,
                bg=self.card_color,
                bd=1,
                relief="solid"
            )
            card.grid(
                row=row,
                column=column,
                padx=6,
                pady=6,
                sticky="nsew"
            )

            tk.Label(
                card,
                text=name,
                font=("Segoe UI", 11, "bold"),
                bg=self.card_color,
                fg=self.text_color
            ).pack(
                anchor="w",
                padx=12,
                pady=(8, 1)
            )

            tk.Label(
                card,
                text=description,
                font=("Segoe UI", 8),
                bg=self.card_color,
                fg=self.secondary_text,
                wraplength=220,
                justify="left"
            ).pack(
                anchor="w",
                padx=12,
                pady=(0, 5)
            )

            # A real tk.Button is used here so the Apply control is
            # always visible and clickable on Windows, independent of
            # the ttk theme being used by the main application.
            apply_button = tk.Button(
                card,
                text="Apply",
                font=("Segoe UI", 9, "bold"),
                bg="#e5e7eb",
                fg=self.text_color,
                activebackground="#d1d5db",
                activeforeground=self.text_color,
                relief="raised",
                bd=1,
                highlightthickness=0,
                cursor="hand2",
                height=1,
                padx=8,
                pady=3,
                command=lambda key=preset_key, win=window: self.apply_preset(key, win)
            )
            apply_button.pack(
                fill="x",
                padx=12,
                pady=(0, 8)
            )

        # --------------------------------------------------------
        # FOOTER
        # --------------------------------------------------------

        footer = tk.Frame(
            window,
            bg=self.card_color,
            height=48
        )
        footer.pack(fill="x")
        footer.pack_propagate(False)

        tk.Label(
            footer,
            text="Presets can be undone with Ctrl+Z and restored with Ctrl+Y.",
            font=("Segoe UI", 8),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            side="left",
            padx=18
        )

        tk.Button(
            footer,
            text="Close",
            font=("Segoe UI", 9),
            command=window.destroy,
            bg="#e5e7eb",
            fg=self.text_color,
            relief="raised",
            bd=1,
            padx=12,
            pady=2,
            cursor="hand2"
        ).pack(
            side="right",
            padx=15,
            pady=7
        )

        # Make sure the preset window is brought in front of ImageLab.
        window.after(50, window.lift)
        window.after(50, window.focus_force)

    def apply_preset(
        self,
        preset_name,
        preset_window=None
    ):

        if self.base_image is None:
            return

        # Save the complete current editor state as one undo step.
        self.commit_slider_history()
        self.record_history(
            f"Applied preset: {preset_name}"
        )

        # Reset all processing controls before applying the preset.
        self.brightness_value = 0
        self.contrast_value = 0
        self.current_filter = "Original"
        self.saturation_value = 0
        self.hue_value = 0
        self.red_value = 0
        self.green_value = 0
        self.blue_value = 0
        self.gamma_value = 1.0
        self.threshold_value = 128
        self.advanced_mode = "None"

        # --------------------------------------------------------
        # PRESET DEFINITIONS
        # --------------------------------------------------------

        if preset_name == "Auto Enhance":

            self.advanced_mode = "Auto Enhance"
            self.contrast_value = 8

        elif preset_name == "Brighten":

            self.brightness_value = 25
            self.contrast_value = 3

        elif preset_name == "Darken":

            self.brightness_value = -25
            self.contrast_value = 3

        elif preset_name == "Vivid":

            self.contrast_value = 12
            self.saturation_value = 28
            self.gamma_value = 1.05

        elif preset_name == "Color Boost":

            self.saturation_value = 40
            self.contrast_value = 5

        elif preset_name == "Black & White":

            self.current_filter = "Grayscale"

        elif preset_name == "Portrait":

            self.brightness_value = 6
            self.contrast_value = -6
            self.saturation_value = 8
            self.gamma_value = 1.05
            self.current_filter = "Soft"

        elif preset_name == "Warm":

            self.red_value = 12
            self.green_value = 4
            self.blue_value = -12
            self.saturation_value = 8

        elif preset_name == "Cool":

            self.red_value = -10
            self.green_value = 2
            self.blue_value = 14
            self.saturation_value = 5

        elif preset_name == "Sharp":

            self.contrast_value = 6
            self.current_filter = "Sharpen"

        elif preset_name == "Soft":

            self.contrast_value = -5
            self.current_filter = "Smooth"

        elif preset_name == "Vintage":

            self.contrast_value = -4
            self.saturation_value = -12
            self.red_value = 8
            self.green_value = 2
            self.blue_value = -8
            self.gamma_value = 1.08

        elif preset_name == "Cinematic":

            self.contrast_value = 18
            self.saturation_value = -8
            self.gamma_value = 1.08

        elif preset_name == "Reset to Original":

            pass

        self.apply_adjustments()

        self._sync_processing_controls()

        self.status.config(
            text=f"Preset applied: {preset_name}"
        )

        if preset_window is not None:

            try:
                preset_window.destroy()
            except tk.TclError:
                pass

    def _sync_processing_controls(self):

        # Keep visible controls synchronized with preset values.
        if self.brightness_slider is not None:

            try:
                self.brightness_slider.set(
                    self.brightness_value
                )
            except tk.TclError:
                pass

        if self.contrast_slider is not None:

            try:
                self.contrast_slider.set(
                    self.contrast_value
                )
            except tk.TclError:
                pass

        for name, value in (
            ("saturation", self.saturation_value),
            ("hue", self.hue_value),
            ("red", self.red_value),
            ("green", self.green_value),
            ("blue", self.blue_value),
            ("gamma", self.gamma_value),
            ("threshold", self.threshold_value)
        ):

            slider = self.advanced_sliders.get(
                name
            )

            if slider is not None:

                try:
                    slider.set(value)
                except tk.TclError:
                    pass

    # ============================================================
    # NON-DESTRUCTIVE PROCESSING STACK
    # ============================================================

    def _add_processing_layer(
        self,
        description,
        state=None
    ):

        if state is None:
            state = self.get_editor_state()

        if state is None:
            return

        layer = {
            "name": description,
            "state": state,
            "enabled": True
        }

        self.processing_layers.append(
            layer
        )

        if len(self.processing_layers) > 30:
            self.processing_layers.pop(0)

        self.update_processing_stack_window()

    def _finalize_pending_processing_layer(self):

        if self.is_restoring_history:
            self.pending_processing_layer_description = None
            return

        description = (
            self.pending_processing_layer_description
        )

        if not description:
            return

        state = self.get_editor_state()

        self._add_processing_layer(
            description,
            state
        )

        self.pending_processing_layer_description = None

    def _processing_layer_summary(self, state):

        if state is None:
            return "No processing state"

        parts = []

        if state.get("brightness", 0) != 0:
            parts.append(
                f"Brightness {state['brightness']:+d}"
            )

        if state.get("contrast", 0) != 0:
            parts.append(
                f"Contrast {state['contrast']:+d}"
            )

        if state.get("saturation", 0) != 0:
            parts.append(
                f"Saturation {state['saturation']:+d}"
            )

        if state.get("hue", 0) != 0:
            parts.append(
                f"Hue {state['hue']:+d}"
            )

        rgb = (
            state.get("red", 0),
            state.get("green", 0),
            state.get("blue", 0)
        )

        if any(value != 0 for value in rgb):
            parts.append(
                "RGB "
                f"{rgb[0]:+d}/"
                f"{rgb[1]:+d}/"
                f"{rgb[2]:+d}"
            )

        gamma = state.get("gamma", 1.0)

        if abs(gamma - 1.0) > 0.001:
            parts.append(
                f"Gamma {gamma:.1f}"
            )

        if state.get("filter", "Original") != "Original":
            parts.append(
                state["filter"]
            )

        if state.get("advanced_mode", "None") != "None":
            parts.append(
                state["advanced_mode"]
            )

        if not parts:
            return "Original / no active adjustments"

        return " • ".join(parts)

    def show_processing_stack(self):

        if self.base_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return

        if self.processing_stack_window is not None:

            try:
                if self.processing_stack_window.winfo_exists():
                    self.processing_stack_window.lift()
                    self.update_processing_stack_window()
                    return
            except tk.TclError:
                pass

        window = tk.Toplevel(
            self.root
        )

        self.processing_stack_window = window

        window.title(
            "ImageLab - Processing Stack"
        )

        window.geometry(
            "850x600"
        )

        window.minsize(
            700,
            500
        )

        window.configure(
            bg=self.bg_color
        )

        window.protocol(
            "WM_DELETE_WINDOW",
            self.close_processing_stack_window
        )

        # Header
        header = tk.Frame(
            window,
            bg=self.card_color,
            height=70
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        tk.Label(
            header,
            text="Processing Stack",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=20,
            pady=(12, 0)
        )

        tk.Label(
            header,
            text=(
                "Non-destructive checkpoints of your editing states. "
                "Your original image is never overwritten."
            ),
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            anchor="w",
            padx=22
        )

        body = tk.Frame(
            window,
            bg=self.bg_color
        )

        body.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        body.grid_columnconfigure(
            0,
            weight=1
        )

        body.grid_columnconfigure(
            1,
            weight=0
        )

        body.grid_rowconfigure(
            0,
            weight=1
        )

        list_card = tk.Frame(
            body,
            bg=self.card_color,
            bd=1,
            relief="solid"
        )

        list_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        tk.Label(
            list_card,
            text="Editing Checkpoints",
            font=("Segoe UI", 11, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(12, 8)
        )

        list_frame = tk.Frame(
            list_card,
            bg=self.card_color
        )

        list_frame.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=(0, 12)
        )

        self.processing_stack_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 9),
            bg="#f9fafb",
            fg=self.text_color,
            selectbackground=self.accent_color,
            selectforeground="white",
            activestyle="none"
        )

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.processing_stack_listbox.yview
        )

        self.processing_stack_listbox.configure(
            yscrollcommand=scrollbar.set
        )

        self.processing_stack_listbox.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.processing_stack_listbox.bind(
            "<<ListboxSelect>>",
            self._show_selected_processing_layer
        )

        controls = tk.Frame(
            body,
            bg=self.card_color,
            width=210,
            bd=1,
            relief="solid"
        )

        controls.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        controls.grid_propagate(False)

        tk.Label(
            controls,
            text="Layer Controls",
            font=("Segoe UI", 11, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(
            anchor="w",
            padx=15,
            pady=(12, 10)
        )

        ttk.Button(
            controls,
            text="Capture Current",
            command=self.capture_processing_layer
        ).pack(
            fill="x",
            padx=15,
            pady=4
        )

        ttk.Button(
            controls,
            text="Apply Selected",
            command=self.apply_selected_processing_layer
        ).pack(
            fill="x",
            padx=15,
            pady=4
        )

        ttk.Button(
            controls,
            text="Delete Selected",
            command=self.delete_selected_processing_layer
        ).pack(
            fill="x",
            padx=15,
            pady=4
        )

        ttk.Button(
            controls,
            text="Clear Stack",
            command=self.clear_processing_stack
        ).pack(
            fill="x",
            padx=15,
            pady=4
        )

        ttk.Separator(
            controls,
            orient="horizontal"
        ).pack(
            fill="x",
            padx=15,
            pady=12
        )

        self.processing_stack_status = tk.Label(
            controls,
            text="Select a checkpoint",
            font=("Segoe UI", 8),
            bg=self.card_color,
            fg=self.secondary_text,
            justify="left",
            wraplength=175,
            anchor="nw"
        )

        self.processing_stack_status.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=5
        )

        footer = tk.Frame(
            window,
            bg=self.card_color,
            height=48
        )

        footer.pack(
            fill="x"
        )

        footer.pack_propagate(False)

        tk.Label(
            footer,
            text=(
                "Tip: Apply Selected restores a saved editing state "
                "without changing the original image."
            ),
            font=("Segoe UI", 8),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(
            side="left",
            padx=18
        )

        ttk.Button(
            footer,
            text="Close",
            command=self.close_processing_stack_window
        ).pack(
            side="right",
            padx=15,
            pady=8
        )

        self.update_processing_stack_window()

    def update_processing_stack_window(self):

        if self.processing_stack_listbox is None:
            return

        try:
            self.processing_stack_listbox.delete(
                0,
                tk.END
            )
        except tk.TclError:
            return

        if not self.processing_layers:

            self.processing_stack_listbox.insert(
                tk.END,
                "No checkpoints yet."
            )

            if self.processing_stack_status is not None:
                self.processing_stack_status.config(
                    text=(
                        "No checkpoints yet.\n\n"
                        "Make an edit or use Capture Current "
                        "to save the current processing state."
                    )
                )

            return

        for index, layer in enumerate(
            self.processing_layers,
            start=1
        ):

            marker = "●" if layer.get("enabled", True) else "○"

            self.processing_stack_listbox.insert(
                tk.END,
                f"{marker} {index:02d}. {layer['name']}"
            )

    def _show_selected_processing_layer(self, event=None):

        if self.processing_stack_listbox is None:
            return

        selection = self.processing_stack_listbox.curselection()

        if not selection:
            return

        index = selection[0]

        if index >= len(self.processing_layers):
            return

        layer = self.processing_layers[index]

        if self.processing_stack_status is not None:

            self.processing_stack_status.config(
                text=(
                    f"Checkpoint {index + 1}\n\n"
                    f"{layer['name']}\n\n"
                    f"{self._processing_layer_summary(layer['state'])}"
                )
            )

    def capture_processing_layer(self):

        if self.base_image is None:
            return

        description = (
            f"Manual checkpoint {len(self.processing_layers) + 1}"
        )

        self._add_processing_layer(
            description,
            self.get_editor_state()
        )

        self.update_processing_stack_window()

        if self.processing_stack_listbox is not None:

            last_index = (
                len(self.processing_layers) - 1
            )

            if last_index >= 0:
                self.processing_stack_listbox.selection_set(
                    last_index
                )
                self.processing_stack_listbox.see(
                    last_index
                )
                self._show_selected_processing_layer()

        self.status.config(
            text="Current processing state captured"
        )

    def apply_selected_processing_layer(self):

        if not self.processing_layers:
            return

        if self.processing_stack_listbox is None:
            return

        selection = self.processing_stack_listbox.curselection()

        if not selection:

            messagebox.showinfo(
                "Processing Stack",
                "Select a checkpoint first."
            )

            return

        index = selection[0]

        if index >= len(self.processing_layers):
            return

        layer = self.processing_layers[index]

        self.commit_slider_history()

        self.record_history(
            f"Restored stack checkpoint: {layer['name']}"
        )

        self.restore_editor_state(
            layer["state"]
        )

        # restore_editor_state is a history operation, so do not create
        # another checkpoint from the pending history description.
        self.pending_processing_layer_description = None

        self.status.config(
            text=f"Processing checkpoint restored: {layer['name']}"
        )

        self.update_processing_stack_window()

    def delete_selected_processing_layer(self):

        if not self.processing_layers:
            return

        if self.processing_stack_listbox is None:
            return

        selection = self.processing_stack_listbox.curselection()

        if not selection:

            messagebox.showinfo(
                "Processing Stack",
                "Select a checkpoint first."
            )

            return

        index = selection[0]

        if index >= len(self.processing_layers):
            return

        removed = self.processing_layers.pop(
            index
        )

        self.update_processing_stack_window()

        self.status.config(
            text=f"Removed checkpoint: {removed['name']}"
        )

    def clear_processing_stack(self):

        if not self.processing_layers:
            return

        if not messagebox.askyesno(
            "Clear Processing Stack",
            "Clear all saved processing checkpoints?\n\n"
            "This does not change your current image or undo history."
        ):
            return

        self.processing_layers.clear()
        self.update_processing_stack_window()

        self.status.config(
            text="Processing stack cleared"
        )

    def close_processing_stack_window(self):

        try:
            self.processing_stack_window.destroy()
        except (
            tk.TclError,
            AttributeError
        ):
            pass

        self.processing_stack_window = None
        self.processing_stack_listbox = None
        self.processing_stack_status = None

    # ============================================================
    # IMAGE COMPARISON / EVALUATION
    # ============================================================

    def _calculate_image_metrics(self):

        if self.original_image is None or self.current_image is None:
            return None

        original = self.original_image.convert("RGB")
        processed = self.current_image.convert("RGB")

        original_size = original.size
        processed_size = processed.size
        normalized = processed.size != original.size

        if normalized:
            processed_for_metrics = processed.resize(
                original.size,
                Image.Resampling.LANCZOS
            )
        else:
            processed_for_metrics = processed

        original_array = np.asarray(original, dtype=np.float32)
        processed_array = np.asarray(processed_for_metrics, dtype=np.float32)
        delta = original_array - processed_array
        difference = np.abs(delta)

        mse = float(np.mean(delta ** 2))
        mae = float(np.mean(difference))
        rmse = float(np.sqrt(mse))
        psnr = float("inf") if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))

        changed_pixels = np.any(difference > 0, axis=2)
        changed_count = int(np.count_nonzero(changed_pixels))
        total_pixels = int(changed_pixels.size)
        changed_percentage = (changed_count / total_pixels * 100) if total_pixels else 0.0

        channel_mae = np.mean(difference, axis=(0, 1))
        channel_rmse = np.sqrt(np.mean(delta ** 2, axis=(0, 1)))
        difference_map = np.mean(difference, axis=2)

        return {
            "original_size": original_size,
            "processed_size": processed_size,
            "normalized": normalized,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "psnr": psnr,
            "changed_count": changed_count,
            "total_pixels": total_pixels,
            "changed_percentage": changed_percentage,
            "channel_mae": channel_mae,
            "channel_rmse": channel_rmse,
            "difference_map": difference_map
        }

    def show_image_evaluation(self):

        if self.original_image is None or self.current_image is None:
            messagebox.showwarning("No Image", "Please open and process an image first.")
            return

        self.commit_slider_history()

        if self.evaluation_window is not None:
            try:
                if self.evaluation_window.winfo_exists():
                    self.update_image_evaluation()
                    self.evaluation_window.lift()
                    return
            except tk.TclError:
                pass

        self.evaluation_window = tk.Toplevel(self.root)
        self.evaluation_window.title("ImageLab - Image Comparison & Evaluation")
        self.evaluation_window.geometry("1150x760")
        self.evaluation_window.minsize(950, 650)
        self.evaluation_window.configure(bg=self.bg_color)
        self.evaluation_window.protocol("WM_DELETE_WINDOW", self.close_image_evaluation)

        header = tk.Frame(self.evaluation_window, bg=self.card_color, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="Image Comparison & Evaluation",
            font=("Segoe UI", 19, "bold"), bg=self.card_color, fg=self.text_color
        ).pack(side="left", padx=20)

        ttk.Button(header, text="Refresh", command=self.update_image_evaluation).pack(side="right", padx=(5, 20))
        ttk.Button(header, text="Export Report", command=self.export_image_evaluation_report).pack(side="right", padx=5)

        content = tk.Frame(self.evaluation_window, bg=self.bg_color)
        content.pack(fill="both", expand=True, padx=15, pady=15)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        visual_card = tk.Frame(content, bg=self.card_color, bd=1, relief="solid")
        visual_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        tk.Label(
            visual_card, text="Difference Map", font=("Segoe UI", 12, "bold"),
            bg=self.card_color, fg=self.text_color
        ).pack(anchor="w", padx=15, pady=(12, 3))

        tk.Label(
            visual_card,
            text="Brighter areas indicate larger pixel differences.",
            font=("Segoe UI", 9), bg=self.card_color, fg=self.secondary_text
        ).pack(anchor="w", padx=15, pady=(0, 8))

        plot_frame = tk.Frame(visual_card, bg=self.card_color)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.evaluation_figure = Figure(figsize=(5, 4), dpi=100)
        self.evaluation_canvas = FigureCanvasTkAgg(self.evaluation_figure, master=plot_frame)
        self.evaluation_canvas.get_tk_widget().pack(fill="both", expand=True)

        metrics_card = tk.Frame(content, bg=self.card_color, bd=1, relief="solid")
        metrics_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        tk.Label(
            metrics_card, text="Evaluation Metrics", font=("Segoe UI", 12, "bold"),
            bg=self.card_color, fg=self.text_color
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.evaluation_report = tk.Text(
            metrics_card, wrap="word", font=("Consolas", 9), bg="#f9fafb",
            fg=self.text_color, relief="flat", padx=12, pady=10, state="disabled"
        )
        scrollbar = ttk.Scrollbar(metrics_card, orient="vertical", command=self.evaluation_report.yview)
        self.evaluation_report.configure(yscrollcommand=scrollbar.set)
        self.evaluation_report.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=(0, 12))

        self.update_image_evaluation()

    def update_image_evaluation(self):

        if self.evaluation_window is None or self.original_image is None or self.current_image is None:
            return

        metrics = self._calculate_image_metrics()
        if metrics is None:
            return

        psnr_text = "∞" if np.isinf(metrics["psnr"]) else f'{metrics["psnr"]:.4f} dB'
        mae = metrics["channel_mae"]
        rmse = metrics["channel_rmse"]

        report = "\n".join([
            "IMAGE COMPARISON",
            "=" * 48,
            f'Original size       : {metrics["original_size"][0]} x {metrics["original_size"][1]}',
            f'Processed size      : {metrics["processed_size"][0]} x {metrics["processed_size"][1]}',
            f'Metric normalization: {"Yes - processed image normalized to original size" if metrics["normalized"] else "No"}',
            "",
            "QUANTITATIVE METRICS",
            "=" * 48,
            f'MSE                 : {metrics["mse"]:.6f}',
            f'MAE                 : {metrics["mae"]:.6f}',
            f'RMSE                : {metrics["rmse"]:.6f}',
            f'PSNR                : {psnr_text}',
            f'Changed pixels      : {metrics["changed_count"]:,}',
            f'Total pixels        : {metrics["total_pixels"]:,}',
            f'Pixels changed      : {metrics["changed_percentage"]:.4f}%',
            "",
            "RGB CHANNEL DIFFERENCE",
            "=" * 48,
            f'Red MAE             : {mae[0]:.6f}',
            f'Green MAE           : {mae[1]:.6f}',
            f'Blue MAE            : {mae[2]:.6f}',
            f'Red RMSE            : {rmse[0]:.6f}',
            f'Green RMSE          : {rmse[1]:.6f}',
            f'Blue RMSE           : {rmse[2]:.6f}',
            "",
            "INTERPRETATION",
            "=" * 48,
            "MAE measures average absolute pixel difference.",
            "MSE emphasizes larger pixel differences.",
            "RMSE is the square root of MSE.",
            "PSNR expresses similarity in decibels.",
            "The difference map shows where changes are concentrated."
        ])

        self.evaluation_report.configure(state="normal")
        self.evaluation_report.delete("1.0", tk.END)
        self.evaluation_report.insert(tk.END, report)
        self.evaluation_report.configure(state="disabled")

        self.evaluation_figure.clear()
        axis = self.evaluation_figure.add_subplot(111)
        diff = metrics["difference_map"]
        maximum = max(1.0, float(np.max(diff)))
        artist = axis.imshow(diff, cmap="inferno", vmin=0, vmax=maximum)
        axis.set_title("Pixel Difference Heatmap")
        axis.set_xlabel("X Position")
        axis.set_ylabel("Y Position")
        self.evaluation_figure.colorbar(
            artist, ax=axis, fraction=0.046, pad=0.04,
            label="Mean absolute RGB difference"
        )
        self.evaluation_figure.tight_layout()
        self.evaluation_canvas.draw_idle()

    def export_image_evaluation_report(self):

        if self.original_image is None or self.current_image is None:
            messagebox.showwarning("No Image", "Please open and process an image first.")
            return

        metrics = self._calculate_image_metrics()
        if metrics is None:
            return

        mae = metrics["channel_mae"]
        rmse = metrics["channel_rmse"]
        psnr_text = "Infinity" if np.isinf(metrics["psnr"]) else f'{metrics["psnr"]:.6f} dB'

        report = "\n".join([
            "ImageLab - Image Comparison & Evaluation Report",
            "=" * 60,
            "",
            f'Original size: {metrics["original_size"][0]} x {metrics["original_size"][1]}',
            f'Processed size: {metrics["processed_size"][0]} x {metrics["processed_size"][1]}',
            f'Metric normalization: {"Yes" if metrics["normalized"] else "No"}',
            "",
            "Quantitative Metrics",
            "-" * 60,
            f'MSE: {metrics["mse"]:.8f}',
            f'MAE: {metrics["mae"]:.8f}',
            f'RMSE: {metrics["rmse"]:.8f}',
            f'PSNR: {psnr_text}',
            f'Changed pixels: {metrics["changed_count"]:,}',
            f'Total pixels: {metrics["total_pixels"]:,}',
            f'Percentage of pixels changed: {metrics["changed_percentage"]:.6f}%',
            "",
            "RGB Channel Metrics",
            "-" * 60,
            f'Red MAE: {mae[0]:.8f}',
            f'Green MAE: {mae[1]:.8f}',
            f'Blue MAE: {mae[2]:.8f}',
            f'Red RMSE: {rmse[0]:.8f}',
            f'Green RMSE: {rmse[1]:.8f}',
            f'Blue RMSE: {rmse[2]:.8f}',
            "",
            "Current Processing",
            "-" * 60,
            f'Brightness: {self.brightness_value}',
            f'Contrast: {self.contrast_value}',
            f'Saturation: {self.saturation_value}',
            f'Hue: {self.hue_value}',
            f'Red adjustment: {self.red_value}',
            f'Green adjustment: {self.green_value}',
            f'Blue adjustment: {self.blue_value}',
            f'Gamma: {self.gamma_value}',
            f'Threshold: {self.threshold_value}',
            f'Advanced mode: {self.advanced_mode}',
            f'Filter: {self.current_filter}',
            "",
            "Generated by ImageLab - Image Processing & Enhancement System"
        ])

        file_path = filedialog.asksaveasfilename(
            title="Export Evaluation Report",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("CSV File", "*.csv")]
        )
        if not file_path:
            return

        try:
            if Path(file_path).suffix.lower() == ".csv":
                import csv
                rows = [
                    ("Original Width", metrics["original_size"][0]),
                    ("Original Height", metrics["original_size"][1]),
                    ("Processed Width", metrics["processed_size"][0]),
                    ("Processed Height", metrics["processed_size"][1]),
                    ("MSE", metrics["mse"]),
                    ("MAE", metrics["mae"]),
                    ("RMSE", metrics["rmse"]),
                    ("PSNR dB", "Infinity" if np.isinf(metrics["psnr"]) else metrics["psnr"]),
                    ("Changed Pixels", metrics["changed_count"]),
                    ("Total Pixels", metrics["total_pixels"]),
                    ("Pixels Changed %", metrics["changed_percentage"]),
                    ("Red MAE", mae[0]),
                    ("Green MAE", mae[1]),
                    ("Blue MAE", mae[2]),
                    ("Red RMSE", rmse[0]),
                    ("Green RMSE", rmse[1]),
                    ("Blue RMSE", rmse[2])
                ]
                with open(file_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Metric", "Value"])
                    writer.writerows(rows)
            else:
                Path(file_path).write_text(report, encoding="utf-8")

            messagebox.showinfo("Report Exported", f"Evaluation report saved to:\n{file_path}")
        except OSError as error:
            messagebox.showerror("Export Error", f"Could not save the report.\n\n{error}")

    def close_image_evaluation(self):

        try:
            self.evaluation_window.destroy()
        except (tk.TclError, AttributeError):
            pass

        self.evaluation_window = None
        self.evaluation_canvas = None
        self.evaluation_figure = None
        self.evaluation_report = None

    # ============================================================
    # SETTINGS
    # ============================================================

    def show_settings(self):

        if self.settings_window is not None:
            try:
                if self.settings_window.winfo_exists():
                    self.settings_window.lift()
                    self.settings_window.focus_force()
                    return
            except tk.TclError:
                pass

        window = tk.Toplevel(self.root)
        self.settings_window = window
        window.title("ImageLab - Settings")
        window.geometry("760x650")
        window.minsize(680, 560)
        window.configure(bg=self.bg_color)
        window.transient(self.root)

        def close_settings():
            self.settings_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_settings)

        header = tk.Frame(window, bg=self.card_color, bd=0)
        header.pack(fill="x")

        tk.Label(
            header,
            text="ImageLab Settings",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_color,
            fg=self.text_color
        ).pack(anchor="w", padx=20, pady=(18, 2))

        tk.Label(
            header,
            text="Configure how ImageLab previews, edits, saves, and manages history.",
            font=("Segoe UI", 9),
            bg=self.card_color,
            fg=self.secondary_text
        ).pack(anchor="w", padx=20, pady=(0, 15))

        body = tk.Frame(window, bg=self.bg_color)
        body.pack(fill="both", expand=True, padx=15, pady=15)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ---------- General ----------
        general = tk.LabelFrame(
            body, text="General", font=("Segoe UI", 10, "bold"),
            bg=self.card_color, fg=self.text_color, bd=1, relief="solid"
        )
        general.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=0)

        self.settings_confirm_reset_var = tk.BooleanVar(value=self.settings_confirm_reset)
        ttk.Checkbutton(
            general,
            text="Confirm before Reset / Clear History",
            variable=self.settings_confirm_reset_var
        ).pack(anchor="w", padx=15, pady=(18, 8))

        tk.Label(
            general, text="Maximum history steps", bg=self.card_color,
            fg=self.text_color, font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=15, pady=(14, 4))

        self.settings_history_var = tk.IntVar(value=self.settings_max_history)
        history_scale = tk.Scale(
            general, from_=10, to=100, orient="horizontal",
            variable=self.settings_history_var, resolution=5,
            bg=self.card_color, fg=self.text_color,
            highlightthickness=0, troughcolor="#dbe3ef"
        )
        history_scale.pack(fill="x", padx=15)

        tk.Label(
            general, text="Undo/Redo and Processing History are limited to this number.",
            bg=self.card_color, fg=self.secondary_text,
            font=("Segoe UI", 8), wraplength=280, justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 15))

        # ---------- Preview ----------
        preview = tk.LabelFrame(
            body, text="Preview & Performance", font=("Segoe UI", 10, "bold"),
            bg=self.card_color, fg=self.text_color, bd=1, relief="solid"
        )
        preview.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=0)

        self.settings_live_preview_var = tk.BooleanVar(value=self.settings_live_preview)
        ttk.Checkbutton(
            preview,
            text="Live preview while adjusting sliders",
            variable=self.settings_live_preview_var
        ).pack(anchor="w", padx=15, pady=(18, 8))

        tk.Label(
            preview, text="Preview quality", bg=self.card_color,
            fg=self.text_color, font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=15, pady=(14, 4))

        self.settings_preview_var = tk.StringVar(
            value=str(self.settings_preview_quality)
        )
        quality_frame = tk.Frame(preview, bg=self.card_color)
        quality_frame.pack(fill="x", padx=15, pady=3)
        for value in (600, 900, 1200, 1600):
            ttk.Radiobutton(
                quality_frame, text=str(value) + " px",
                variable=self.settings_preview_var, value=str(value)
            ).pack(anchor="w", pady=2)

        tk.Label(
            preview,
            text="Higher values give a more detailed live preview but may use more processing time.",
            bg=self.card_color, fg=self.secondary_text,
            font=("Segoe UI", 8), wraplength=280, justify="left"
        ).pack(anchor="w", padx=15, pady=(8, 15))

        # ---------- Saving ----------
        saving = tk.LabelFrame(
            body, text="Saving", font=("Segoe UI", 10, "bold"),
            bg=self.card_color, fg=self.text_color, bd=1, relief="solid"
        )
        saving.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(12, 0))

        tk.Label(
            saving, text="Default save format", bg=self.card_color,
            fg=self.text_color, font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        self.settings_format_var = tk.StringVar(value=self.settings_default_format)
        ttk.Combobox(
            saving, textvariable=self.settings_format_var,
            values=("PNG", "JPEG", "BMP", "TIFF"),
            state="readonly", width=18
        ).pack(anchor="w", padx=15)

        tk.Label(
            saving, text="JPEG quality", bg=self.card_color,
            fg=self.text_color, font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=15, pady=(12, 4))

        self.settings_jpeg_var = tk.IntVar(value=self.settings_jpeg_quality)
        tk.Scale(
            saving, from_=50, to=100, orient="horizontal",
            variable=self.settings_jpeg_var, resolution=1,
            bg=self.card_color, fg=self.text_color,
            highlightthickness=0, troughcolor="#dbe3ef"
        ).pack(fill="x", padx=15, pady=(0, 8))

        # ---------- Current status ----------
        status_card = tk.LabelFrame(
            body, text="Current Editor Status", font=("Segoe UI", 10, "bold"),
            bg=self.card_color, fg=self.text_color, bd=1, relief="solid"
        )
        status_card.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(12, 0))

        status_text = (
            f"Brightness: {self.brightness_value:+d}\n"
            f"Contrast: {self.contrast_value:+d}\n"
            f"Saturation: {self.saturation_value:+d}\n"
            f"Hue: {self.hue_value:+d}\n"
            f"RGB: {self.red_value:+d} / {self.green_value:+d} / {self.blue_value:+d}\n"
            f"Gamma: {self.gamma_value:.1f}\n"
            f"Filter: {self.current_filter}\n"
            f"Advanced: {self.advanced_mode}\n"
            f"Zoom: {'Fit' if self.zoom_mode == 'fit' else f'{self.zoom_level * 100:.0f}%'}\n"
            f"Undo steps: {len(self.undo_stack)}\n"
            f"Redo steps: {len(self.redo_stack)}"
        )
        tk.Label(
            status_card, text=status_text, bg=self.card_color,
            fg=self.text_color, font=("Consolas", 9), justify="left"
        ).pack(anchor="w", padx=15, pady=15)

        # ---------- Bottom buttons ----------
        buttons = tk.Frame(window, bg=self.bg_color)
        buttons.pack(fill="x", padx=15, pady=(0, 15))

        def restore_defaults():
            self.settings_history_var.set(30)
            self.settings_preview_var.set("900")
            self.settings_live_preview_var.set(True)
            self.settings_confirm_reset_var.set(True)
            self.settings_format_var.set("PNG")
            self.settings_jpeg_var.set(95)

        def apply_settings():
            self.settings_max_history = max(10, int(self.settings_history_var.get()))
            self.settings_preview_quality = int(self.settings_preview_var.get())
            self.settings_live_preview = bool(self.settings_live_preview_var.get())
            self.settings_confirm_reset = bool(self.settings_confirm_reset_var.get())
            self.settings_default_format = self.settings_format_var.get()
            self.settings_jpeg_quality = int(self.settings_jpeg_var.get())

            self.preview_max_dimension = self.settings_preview_quality

            while len(self.undo_stack) > self.settings_max_history:
                self.undo_stack.pop(0)
            while len(self.history) > self.settings_max_history:
                self.history.pop(0)

            self._preview_base_image = None
            self._preview_source_id = None
            self.status.config(text="Settings applied")
            close_settings()

        # Use regular Tk buttons here so the button text remains visible
        # regardless of the active Windows/ttk theme.
        tk.Button(
            buttons,
            text="Restore Defaults",
            command=restore_defaults,
            bg=self.card_color,
            fg=self.text_color,
            activebackground="#e5e7eb",
            activeforeground=self.text_color,
            relief="raised",
            bd=1,
            padx=12,
            pady=5,
            font=("Segoe UI", 9)
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            buttons,
            text="Cancel",
            command=close_settings,
            bg=self.card_color,
            fg=self.text_color,
            activebackground="#e5e7eb",
            activeforeground=self.text_color,
            relief="raised",
            bd=1,
            padx=12,
            pady=5,
            font=("Segoe UI", 9)
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            buttons,
            text="Apply Settings",
            command=apply_settings,
            bg=self.card_color,
            fg=self.text_color,
            activebackground="#e5e7eb",
            activeforeground=self.text_color,
            relief="raised",
            bd=1,
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold")
        ).pack(side="right")


# ================================================================
# START APPLICATION
# ================================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ImageLabApp(
        root
    )

    root.mainloop()