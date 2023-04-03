# M_Sum
Multilingual Summary

Powered by Minggz

## Requirements 
Listed at requirements.txt

## Extractive Sumarizer
Using Sentence Ranking method

Our approach : BERT (get the embedding of sentences) + Centroid + Cosine + Sentence Position


Pretrained Bert could be found at [huggingface](https://huggingface.co/models)
## Abstractive Sumarizer
Config the sumy library [sumy](https://pypi.org/project/sumy/)

## Usage

### Install packages
```bash
python -m pip install -r requirements.txt
```

### Extractive
```bash
python src/ext_sum_onnx.py
```

### Abstractive
```bash
python src/abs_sum_onnx.py
```
### Demo simple app by Streamlit
```bash
python -m streamlit run src/app.py
```