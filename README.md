# Tikhonov Relaxation Spectra Analysis

## Purpose

This Python script analyzes time-dependent relaxation data to identify relaxation modes using Tikhonov regularization. It includes automated selection of the regularization parameter through L-curve analysis. The script is adapted from established numerical methods and is intended for use by experimental researchers without prior coding experience.

The Python script that was made for this work is built in such a manner that settings can be changed intuitively when an integrated development environment (IDE) is used. For operation of the script, no prior knowledge is required about Python or the underlying algorithms. 

For more background information on the method and especially the limits of this script, refer to the associated article and its supporting information: 

> Wink, R. *et al.* "Title of the Paper." Journal Name, Volume(Issue), Year. [DOI or URL]

## Requirements

- Python 3.x (Anaconda recommended)
- Spyder IDE (or any Python IDE of choice)
- Required packages:
  - numpy
  - matplotlib
  - scipy

All required packages are included by default in the Anaconda distribution.

## File Structure

Download the `tikhonov_relaxation_spectrum.py` script from this GitHub, and place the script and data files in a structured folder layout. For example:

```
your_project_folder/
├── tikhonov_relaxation_spectrum.py
├── data/
│   └── input_data.txt
```

The script uses **relative file paths**, meaning file locations are interpreted with respect to the folder where the script is run.

## Usage Instructions

1. Open the script in Spyder.
2. At the top of the script, modify only the following settings:

```python
data_input  = 'data\\input_data.txt'
data_output = 'data\\export_data.txt'

export_lcurve = True
```

- `data_input`: Path to the input file (TXT format with two columns; see below).
- `data_output`: Path where the output (relaxation spectrum) will be saved.
- `export_lcurve`: Set to `True` to also export the L-curve data and selected regularization parameter. Otherwise, set to `False`

Note that any name could be used for the names of the files, to the user's discretion.

3. Run the script (press F5 or use the Run button). 

## Input Format

The input file should be a plain text file with two columns:

- First column: time values (e.g. in seconds)
- Second column: corresponding experimental data

No header row should be included.

## Output

  *The plots will appear in the Spyder plot viewer. It is not exported unless manually saved.*
- A plot of the computed relaxation spectrum
- A plot of the L-curve with the selected corner
- Output text file containing the relaxation spectrum (`data_output`; here `export_data.txt`). This contains two columns, the relaxation times and their respective relaxation strengths. Note that the relaxation times will be in the same unit as the provided time data.
- If enabled, a second file with the L-curve data will be created containing three columns (here `export_data_lcurve.txt`). This contains:
  - On the first line: the optimal regression parameter,  the residual- and the solution norm at the corner of the L-curve.
  - The rest of the file: all 200 tested regularization parameters, and the found residual- and the solution norms. 
- The optimal regularization parameter and the peak in relaxation time are also printed to the console.

## Notes

- The script uses singular value decomposition and Tikhonov regularization. Familiarity with these methods is not required, but it is helpful for interpreting the results.
- Make sure the input data is properly formatted and does not contain missing or malformed values. Initial measurement noise (e.g. due to step-raise) must be deleted. The script is agnostic against normalization of the data.

## Author

Roy Wink  
Eindhoven University of Technology, 2025

## License

This project is licensed under the **GNU General Public License v3.0**.

You are free to use, study, share and modify the code under the following conditions:

- Any distributed version of the software must also be licensed under GPL v3.
- If you modify the code and distribute it, you must also share the source code.
- The original copyright notice and license must be preserved.

For the full license text, see the LICENCE.md file, or  
https://www.gnu.org/licenses/gpl-3.0.html

Users of the code are kindly requested to cite the present work when employing the code in scientific work.