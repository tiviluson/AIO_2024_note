# Transfer Learning

# Knowledge distillation

# Object detection
![alt text](image-18.png)

## Non Maximal Suppression (NMS)
![alt text](image-9.png)
## With CNN
![alt text](image-5.png)
![alt text](image-8.png)
![alt text](image-7.png)

## R-CNN and Fast R-CNN

## With YOLOs
![alt text](image-19.png)

### YOLO-v1
**Architecture**: divides image into 7×7 grid, each cell predicts 2 boxes + 20 classes → 1470 output vector.
![alt text](image-20.png)
**Loss function**: 
Loss = bbox regression + object confidence + class prediction (with weighted terms).
![alt text](image-22.png)

### YOLO-v2
**Architecture**: adds anchor boxes, removes FC layers, uses lighter backbone (Darknet-19).
![alt text](image-27.png)
**Loss function**: same as YOLO-v1, but with additional terms for bounding box regression
![alt text](image-23.png)
![alt text](image-24.png)

### YOLO-v3
**Architecture**: adds SPP (Spatial Pyramid Pooling) module, uses Darknet-53 backbone.
![alt text](image-25.png)

## With Transformers

# Mixture of Experts (MoE)

# NLP preprocessing
## Preprocessing 
(TODO): clean up code
[Tokenization_and_Embedding.ipynb](Tokenization_and_Embedding.ipynb)
[Tokenizer playground](https://xenova-the-tokenizer-playground.static.hf.space/index.html)
[Tokenization in Hugging Face](https://huggingface.co/docs/transformers/fast_tokenizers)

## Stemming and lemmatization (optional)
- Stemming: reduces words to their root form (e.g., "running" to "run")
- Lemmatization: reduces words to their base form (e.g., "better" to "good")


# Part-of-Speech (POS) tagging
![alt text](image.png)
![alt text](image-1.png)


# Named Entity Recognition (NER)
![alt text](image-10.png)
![alt text](image-11.png)

# Aspect-based Sentiment Analysis
![alt text](image-3.png)
![alt text](image-2.png)
![alt text](image-4.png)

# BERTs

# Text generation

# Machine Translation

# Question Answering (QA)
![alt text](image-13.png)
## By classification
![alt text](image-14.png)
## By extraction
![alt text](image-16.png)
![alt text](image-12.png)
## By generation
![alt text](image-15.png)

# KAN

# Counting with YOLO

# Face detection and identification