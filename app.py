import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS

# Initialize Flask application
app = Flask(__name__)

CORS(app)
# Define model and tokenizer paths
MODEL_PATH = "model"  # Path to the folder containing model.safetensors, tokenizer files, etc.

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on: {device}")

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

model.to(device)

# Generate answer function
def generate_answer(question, max_length=2048):
    """
    Generate an answer for a given question.
    """
    # Format the input for T5
    input_text = f"question: {question}"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=max_length,
            min_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.1
        )
    
    # Decode the output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# API endpoint for generating answers
@app.route("/answer", methods=["POST"])
def get_answer():
    try:
        # Parse the request JSON
        data = request.json
        question = data.get("question", "")
        
        if not question:
            return jsonify({"error": "Question field is required."}), 400
        
        # Generate the answer
        answer = generate_answer(question)
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
