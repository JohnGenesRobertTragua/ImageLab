# ImageLab

**ImageLab** is a desktop **Image Processing & Enhancement System** developed in Python. It provides a graphical interface for opening, editing, enhancing, analyzing, comparing, and batch-processing images.

The project was built as a portfolio application to demonstrate Python GUI development, image processing, numerical image manipulation, data analysis, and desktop application design.

## Features

### Image Management
- Open and import common image formats
- Save processed images
- Reset the editor to the original image
- Support for PNG, JPEG, BMP, TIFF, and other Pillow-supported formats
- Image information and metadata display

### Basic Image Adjustments
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Hue adjustment
- RGB channel adjustment
- Gamma correction

### Image Filters
- Grayscale
- Sepia
- Negative / Invert
- Blur
- Sharpen
- Edge Enhance
- Emboss
- Smooth
- Contour

### Advanced Image Processing
- Histogram Equalization
- Auto Enhance
- Thresholding
- NumPy-based pixel processing
- Pillow-based image filtering and enhancement

### Image Transformations
- Horizontal flip
- Vertical flip
- 90°, 180°, and 270° rotation
- Custom rotation
- Visual crop
- Resize
- Reset transformations

### Editing Workflow
- Undo and Redo
- Processing History
- Processing Stack
- Before/After comparison
- Presets
- Zoom controls
- Active editing-tool indication
- Live editing preview

### Image Analysis
- Image dimensions
- Pixel count
- Image mode and channels
- Minimum, maximum, and mean pixel values
- RGB channel statistics
- RGB histograms
- Processing information
- Metadata / EXIF information when available

### Image Comparison
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- Changed pixel count and percentage
- RGB channel error statistics
- Difference heatmap
- Comparison report export

### Batch Processing
- Add multiple images
- Add an image folder
- Apply image processing settings to multiple images
- Batch brightness, contrast, saturation, hue, RGB, and gamma adjustments
- Apply filters and advanced processing
- Batch resizing
- Select output format and output folder
- Processing progress and result summary

## Presets

ImageLab includes predefined editing presets that provide quick starting configurations, including:

- Auto Enhance
- Brighten
- Darken
- Vivid
- Color Boost
- Black & White
- Portrait
- Warm
- Cool
- Sharp
- Soft
- Vintage
- Cinematic
- Reset to Original

Presets can be used as a starting point and then fine-tuned with the individual editing controls.

## Technologies Used

- **Python**
- **Tkinter / ttk** — desktop graphical user interface
- **Pillow (PIL)** — image loading, transformation, filtering, and saving
- **NumPy** — numerical and pixel-level image processing
- **Matplotlib** — histogram visualization and image analysis

## Project Structure

```text
ImageLab/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
├── data/
└── output/
```

> Some folders may be empty or created as the application is used.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JohnGenesRobertTragua/ImageLab.git
```

### 2. Open the project

```bash
cd ImageLab
```

### 3. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the required packages

```powershell
pip install -r requirements.txt
```

## Run the Application

Start ImageLab with:

```powershell
python main.py
```

The ImageLab desktop application will open and can then be used to import and process images.

## Requirements

The project dependencies are listed in `requirements.txt`:

```text
numpy
Pillow
matplotlib
```

Python 3.12 was used during development.

## GitHub

Repository:

https://github.com/JohnGenesRobertTragua/ImageLab

## Purpose

This project demonstrates practical development of a Python desktop application with:

- GUI design
- Event-driven programming
- Image processing
- NumPy array manipulation
- Data visualization
- File handling
- Undo/Redo state management
- Batch processing
- Image analysis
- Application settings
- Git and GitHub version control

## Author

**John Genes Robert Tragua**

Computer Engineering Student  
Software Development Portfolio Project
