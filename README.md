# Find Orientation in Soccer Field

## Description

Final Project for CM-203 Computer Vision course. The idea is to identify the side/orientation in the soccer field by the background.

## Run

In the root folder, run `pip install -r requirements.txt` to install the packages needed. It's recommend to do this inside a python virtual environment. Besides that, you need to have Eigen installed.

After that, run the `main.py` file inside the `src` folder. The result histogram should appear on the screen and some metrics in the terminal.

You can choose the method to calculate the orientation inside the `main.py` file, by commenting/uncommenting the line that declare the `orient_method` variable.

## Test Dataset

<img src=docs/field.png width=500>

## Examples

The following image shows all matching points between the reference and the test image. Qualitatively, we can see that there are not many outliers.

Using reference images in 45º intervals, we had the results bellow.

In this case, the test angle was 300º and it matched more with the 315º reference, estimating an angle of 333º with the most crude method.

<img src=docs/ref+test_45-45_all.png width=500>

The images bellow show about 2% of the matches, with the green points corresponding to the reference image and the red ones to the test image.

<img src=docs/ref_img_45-45.png width=500>
<img src=docs/test_img_45-45.png width=500>
<img src=docs/ref+test_45-45.png width=500>

Using reference images in 90º intervals, we had the results bellow.

In this case, the test angle was 300º and it matched more with the 270º reference, estimating an angle of 270º with the most crude method.

<img src=docs/ref+test_90-90_all.png width=500>

The images bellow show about 2% of the matches, with the green points corresponding to the reference image and the red ones to the test image.

<img src=docs/ref_img_90-90.png width=500>
<img src=docs/test_img_90-90.png width=500>
<img src=docs/ref+test_90-90.png width=500>


