from io import BytesIO
from typing import List, Tuple

import PIL.Image
import qrcode
from PIL import Image
from fpdf import FPDF
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import GappedSquareModuleDrawer


def generate_pdf(mode: str, story_idea: str, generated_images: List[Tuple[Image.Image, str]]) -> bytes:
    """
    Generate a single-page landscape PDF:
    - Left half: Cover with title, subtitle, story idea
    - Right half: 4 images in 2x2 grid with captions

    Args:
        mode (str): "illustration" or "tshirt"
        story_idea (str): The story or design idea text
        generated_images (List[Tuple[Image.Image, str]]): List of (PIL.Image, caption)

    Returns:
        bytes: The generated PDF as bytes
    """
    pdf = FPDF(orientation='L', unit='mm', format='A4')  # Landscape A4
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Dimensions for layout
    page_width, page_height = 297, 210
    left_width = page_width / 2  # 148.5 mm
    right_width = page_width / 2
    margin = 0

    # Left section (cover)
    draw_cover(pdf, x=margin, y=margin, width=left_width - margin * 2, mode=mode, story_idea=story_idea)

    # Right section (images in 2x2 grid)
    image_area_x = left_width
    image_area_y = margin
    image_area_width = right_width - margin
    image_area_height = page_height - margin * 2

    draw_image_grid(pdf, start_x=image_area_x, start_y=image_area_y,
                    width=image_area_width, height=image_area_height,
                    generated_images=generated_images)

    # Output PDF as bytes
    pdf_bytes = BytesIO(pdf.output())
    pdf_bytes.seek(0)
    return pdf_bytes.read()


def draw_cover(pdf: FPDF, x: float, y: float, width: float, mode: str, story_idea: str) -> None:
    """
    Draw the left half: cover title, subtitle, and story idea.

    Args:
        pdf (FPDF): The PDF object
        x (float): X position
        y (float): Y position
        width (float): Width of the section
        mode (str): The mode ("illustration" or "tshirt")
        story_idea (str): The story idea text
    """
    # Background rectangle
    pdf.set_fill_color(31, 31, 46)
    pdf.rect(x, y, width, 210 - y * 2, 'F')

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(106, 17, 203)
    pdf.set_xy(x, y + 30)
    title = "Your Visual Story" if mode == "illustration" else "Your T-Shirt Designs"
    pdf.multi_cell(width, 12, title, align='C')

    # Subtitle
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.ln(5)
    pdf.set_x(x)
    pdf.multi_cell(width, 10, "Crafted with OpenVINO Toolkit", align='C')

    # Story idea
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(200, 200, 200)
    pdf.ln(10)
    pdf.set_x(x)
    pdf.multi_cell(width, 8, f'Story Idea:\n"{story_idea}"', align='C')

    # QR code
    qr_code = get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/ai_ref_kits/multimodal_ai_visual_generator")
    pdf.image(qr_code, x + (width - 50) / 2, 210 - y - 60, w=50, h=50)


def draw_image_grid(pdf: FPDF, start_x: float, start_y: float, width: float, height: float,
                    generated_images: List[Tuple[Image.Image, str]]) -> None:
    """
    Draws a 2x2 grid of images with captions in the right half, maintaining aspect ratio.
    """
    rows, cols = 2, 2
    cell_width = width / cols
    cell_height = height / rows
    padding = 5
    caption_height = 10

    images = generated_images[:4]  # Take only first 4 images

    for i, (img, caption) in enumerate(images):
        row = i // cols
        col = i % cols

        # Cell position
        x = start_x + col * cell_width
        y = start_y + row * cell_height

        # Available space for the image inside the cell
        max_width = cell_width - padding * 2
        max_height = cell_height - caption_height - padding * 2

        # Get original size and compute scale factor
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height

        if max_width / max_height > aspect_ratio:
            # Fit by height
            display_height = max_height
            display_width = display_height * aspect_ratio
        else:
            # Fit by width
            display_width = max_width
            display_height = display_width / aspect_ratio

        # Center the image in the cell
        img_x = x + (cell_width - display_width) / 2
        img_y = y + padding

        # Place image
        pdf.image(img, img_x, img_y, w=display_width, h=display_height)

        # Caption
        pdf.set_xy(x, y + 2 * padding + display_height)
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(cell_width, 5, caption, align='C')


def get_qr_code(text: str) -> Image:
    qr = qrcode.QRCode(box_size=10, border=2, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(image_factory=StyledPilImage, module_drawer=GappedSquareModuleDrawer())

    return img.resize((256, 256), resample=PIL.Image.LANCZOS)
