# Stable Diffusion with Segment Anything Model

## How to run?
```
git clone https://github.com/angadbawa/Oracle.git
cd stable-diffusion-with-sam-main/
pip install -r requirements.txt
python app.py
```

- To get the mask, we need to click on some part of the image for which you want to mask. If you didn't get the mask as expected, click at different locations, because Segment Anything Model works conditioned on the points of interests.

## Output

![output](./resources/output.jpg)
