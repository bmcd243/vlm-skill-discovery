# DLSD (Discovery) with InternVL

This project implements **Diverse Language-based Skill Discovery**.
It uses Asynchronous VLM labeling to reward an agent for performing behaviors that are *distinguishable* via natural language.

## Architecture
- **Skill Conditioned SAC:** The Policy takes state `s` and skill `z`.
- **Discriminator:** Tries to predict `z` given the VLM text embedding of the video `E(o)`.
- **Reward:** `log P(z | E(o))` (Mutual Information).

## Setup
`pip install -r requirements.txt`
`python src/main.py`