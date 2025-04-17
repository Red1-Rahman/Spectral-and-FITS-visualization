from django import forms

class SpectralImageForm(forms.Form):
    spectral_image = forms.ImageField(label='Upload Spectral Image', required=False,
                                      widget=forms.FileInput(attrs={'class': 'form-control'}))
    calibration_coefficients = forms.CharField(label='Wavelength Calibration (a, b)',
                                               max_length=50,
                                               required=False,
                                               widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 0.1, 400'}))

class FITSFileForm(forms.Form):
    fits_file = forms.FileField(label='Upload FITS File', required=False,
                                widget=forms.FileInput(attrs={'class': 'form-control'}))