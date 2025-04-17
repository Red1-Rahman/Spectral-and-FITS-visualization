from django.shortcuts import render
from django.views import View
from .forms import SpectralImageForm, FITSFileForm
import numpy as np
import cv2
from scipy.signal import find_peaks
from astropy.io import fits
import matplotlib.pyplot as plt
import io
import base64

# Define known spectral lines (in nanometers)
spectral_lines = {
    'H_alpha': 656.3,  # Hydrogen alpha
    'H_beta': 486.1,   # Hydrogen beta
    'H_gamma': 434.0,  # Hydrogen gamma
    'Na_D1': 589.6,    # Sodium D1
    'Na_D2': 589.0,    # Sodium D2
    'Ca_K': 393.4,     # Calcium K
    'Ca_H': 396.8,     # Calcium H
    # Add more spectral lines as needed
}

def pixel_to_wavelength(pixel_position, calibration_coefficients):
    try:
        a, b = map(float, calibration_coefficients.split(','))
        wavelength = a * pixel_position + b
        return wavelength
    except ValueError:
        return None

def map_peak_to_physical_property(peak_wavelength):
    closest_line = None
    min_diff = float('inf')
    tolerance = 5.0

    if peak_wavelength is None:
        return "Invalid wavelength for peak."

    for line, wavelength in spectral_lines.items():
        diff = abs(peak_wavelength - wavelength)
        if diff < min_diff:
            min_diff = diff
            closest_line = line

    if closest_line and min_diff <= tolerance:
        insight = f"Detected emission line at approximately {peak_wavelength:.1f} nm, likely corresponding to {closest_line}."
        if closest_line.startswith('H_'):
            insight += " This indicates the presence of hydrogen, likely in an excited state."
        elif closest_line.startswith('Na_'):
            insight += " This indicates the presence of sodium."
        elif closest_line.startswith('Ca_'):
            insight += " This indicates the presence of calcium."
        return insight
    else:
        return f"Detected peak at {peak_wavelength:.1f} nm, no close match found in the known spectral lines."

def process_spectral_image(image_file, calibration_coefficients):
    try:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.COLOR_BGR2GRAY)
        flattened_image = image.flatten()
        peaks, _ = find_peaks(flattened_image, height=np.mean(flattened_image) + 0.5 * np.std(flattened_image))

        insights = []
        detected_wavelengths = []
        plot_base64 = None

        if peaks is not None and len(peaks) > 0 and calibration_coefficients:
            plt.figure(figsize=(12, 6))
            plt.plot(flattened_image, label="Spectral Data")
            plt.plot(peaks, flattened_image[peaks], "rx", label="Detected Peaks")
            for peak_pixel in peaks:
                wavelength = pixel_to_wavelength(peak_pixel, calibration_coefficients)
                if wavelength:
                    detected_wavelengths.append(wavelength)
                    insight = map_peak_to_physical_property(wavelength)
                    insights.append(insight)
                    plt.annotate(f"{wavelength:.1f} nm",
                                 (peak_pixel, flattened_image[peak_pixel]),
                                 textcoords="offset points",
                                 xytext=(0, 10),
                                 ha='center')
            plt.title("Spectral Image with Detected Peaks and Wavelengths")
            plt.xlabel("Pixel Position")
            plt.ylabel("Intensity")
            plt.legend()
            plt.grid(True)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
        elif not calibration_coefficients:
            insights.append("Please provide wavelength calibration coefficients.")
        else:
            insights.append("No significant peaks detected in the spectral image.")

        return plot_base64, "\n".join(insights)

    except Exception as e:
        return None, f"Error processing spectral image: {e}"

def process_fits_file(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            print("HDU Info:")
            hdul.info()
            image_data = hdul[0].data
            print(f"Min FITS data value: {np.min(image_data)}")
            print(f"Max FITS data value: {np.max(image_data)}")
            print(f"FITS data type: {image_data.dtype}")

            if image_data is not None:
                normalized_data = (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data))
                if np.isnan(normalized_data).all():
                    normalized_data = np.zeros_like(image_data, dtype=float)
            else:
                normalized_data = np.zeros((100, 100), dtype=float) # Create a dummy black image

        plt.figure(figsize=(8, 8))
        plt.imshow(normalized_data, cmap='inferno', origin='lower')
        plt.colorbar(label="Normalized Intensity")
        plt.title("FITS Image Visualization")
        plt.xlabel("Pixel Column")
        plt.ylabel("Pixel Row")

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return plot_base64, "FITS file visualized."

    except Exception as e:
        return None, f"Error processing FITS file: {e}"

class AnalyzerView(View):
    spectral_form_class = SpectralImageForm
    fits_form_class = FITSFileForm
    template_name = 'analyzer/analyzer.html'

    def get(self, request, *args, **kwargs):
        spectral_form = self.spectral_form_class()
        fits_form = self.fits_form_class()
        return render(request, self.template_name, {'spectral_form': spectral_form, 'fits_form': fits_form})

    def post(self, request, *args, **kwargs):
        spectral_form = self.spectral_form_class(request.POST, request.FILES)
        fits_form = self.fits_form_class(request.POST, request.FILES)
        spectral_plot = None
        spectral_insights = None
        fits_plot = None
        fits_status = None

        if spectral_form.is_valid():
            spectral_image = spectral_form.cleaned_data['spectral_image']
            calibration = spectral_form.cleaned_data['calibration_coefficients']
            if spectral_image and calibration:
                spectral_plot, spectral_insights = process_spectral_image(spectral_image, calibration)
            elif spectral_image:
                spectral_insights = "Please provide wavelength calibration coefficients for spectral image analysis."

        if fits_form.is_valid():
            fits_file = fits_form.cleaned_data['fits_file']
            if fits_file:
                fits_plot, fits_status = process_fits_file(fits_file)

        return render(request, self.template_name, {
            'spectral_form': spectral_form,
            'fits_form': fits_form,
            'spectral_plot': spectral_plot,
            'spectral_insights': spectral_insights,
            'fits_plot': fits_plot,
            'fits_status': fits_status,
        })