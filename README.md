## 🧠 ASL Translation using Seq2Seq (LSTM)

This project implements a Sequence-to-Sequence (Seq2Seq) model using LSTM layers in TensorFlow/Keras to translate English text into Sign Language Gloss.
It follows a modular pipeline for data preprocessing, model building, training, and evaluation.

## 📁 Project Structure

ASL-Translation/
│
├── main.py                  # Main execution script
├── model.py                 # Defines Seq2Seq model and training function
├── pipeline.py              # Handles dataset loading and preprocessing
├── dataset_prep.py          # Prepares and splits dataset
├── text_processing.py       # Text cleaning and tokenization utilities
├── aslg_pc12.csv            # Dataset file (English ↔ Gloss)
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files/folders for Git
└── README.md                # Project documentation

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

git clone https://github.com/your-username/asl-translation.git
cd asl-translation

### 2️⃣ Create and activate a virtual environment

For Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

For Windows:
python -m venv .venv
.venv\Scripts\activate

### 3️⃣ Install dependencies

Make sure you have Python 3.10+ installed.

pip install -r requirements.txt

### 4️⃣ Add your dataset

Place your dataset file (e.g. aslg_pc12.csv) in the project root or in a data/ folder.

If it's in a subfolder, update the path in dataset_prep.py or pipeline.py:

data_path = os.path.join(os.getcwd(), 'data', 'aslg_pc12.csv')

### 5️⃣ Run the training pipeline

Execute the main script:

python main.py

## 🧩 Model Architecture

The model uses a Sequential LSTM-based encoder-decoder:

Embedding Layer (input representation)

Encoder LSTM (256, then 128 units)

RepeatVector (sequence context)

Decoder LSTM (128, then 256 units)

TimeDistributed Dense layer with softmax activation for token prediction