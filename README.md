# An Emotion-Aware Support Chatbot

This repository contains the implementation of an emotion-aware mental health support chatbot developed as part of an academic project.

## Features
- Emotion detection with confidence scores
- Confidence-aware response logic
- Crisis keyword detection
- Local LLM (Gemma-2B)
- Flask-based web interface
- Quantitative and qualitative evaluation
- Multilingual support (English & German)
- Fully offline and privacy-preserving

## Structure of the code
An-Emotion-aware-Support-Chatbot/
├── chatbot.py
├── templates/
│ └── index.html
├── evaluate_emotion_model.py
├── qualitative_evaluation.py
├── Emotion_test_data.csv
├── SoSci_data_Survey.csv
├── user_study_group_scores.csv
├── user_study_group_scores.png
├── README.md
└── requirements.txt

## How to Run
```bash
pip install -r requirements.txt
python chatbot.py
```
Then open: http://127.0.0.1:5000

## Ethical Notice
This chatbot provides non-clinical emotional support only and does not offer medical advice.

---
