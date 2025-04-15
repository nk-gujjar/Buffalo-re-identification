# # app.py

# from fastapi import FastAPI, File, Form, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import uvicorn
# from add_info import save_to_mongodb

# app = FastAPI()

# # Allow CORS for frontend communication
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Set to frontend URL in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/upload/")
# async def upload_data(
#     img: UploadFile = File(...),
#     id: str = Form(""),
#     gender: str = Form(""),
#     age: str = Form(""),
#     type: str = Form(""),
#     description: str = Form(""),
#     feature_vector: str = Form(...)  # Serialized numpy array (JSON string)
# ):
#     # Read the image binary
#     image_bytes = await img.read()

#     # Convert JSON string to numpy array
#     feature_vector_np = np.array(eval(feature_vector))

#     # Metadata dictionary
#     item = {
#         "id": id,
#         "gender": gender,
#         "age": age,
#         "type": type,
#         "description": description,
#     }

#     doc_id = save_to_mongodb(image_bytes, feature_vector_np, item)

#     return {"status": "success", "document_id": doc_id}


import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from add_info import add_info, mongodb
from extract_feature import extract_features
import io

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get configuration from environment variables
PORT = int(os.getenv("PORT", 5000))
SECRET_KEY = os.getenv("SECRET_KEY")

# Set Flask secret key
app.secret_key = SECRET_KEY

@app.route('/api/add', methods=['POST'])
def api_add_item():
    """API endpoint to add an item to the database"""
    try:
        # Get form data and file
        item_id = request.form.get('id', '')
        gender = request.form.get('gender', '')
        age = request.form.get('age', '')
        item_type = request.form.get('type', '')
        description = request.form.get('description', '')
        
        # Get image file
        if 'img' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        img_file = request.files['img']
        img_data = img_file.read()
        
        # Prepare data for adding to database
        data = {
            'id': item_id,
            'gender': gender,
            'age': age,
            'type': item_type,
            'description': description,
            'img': img_data
        }
        
        # Add item to database
        result_id = add_info(data)
        
        return jsonify({
            "success": True,
            "message": "Item added successfully",
            "item_id": result_id
        }), 201
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/search-similar', methods=['POST'])
def api_search_similar():
    """API endpoint to search for similar items based on an image"""
    try:
        if 'img' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        img_file = request.files['img']
        img_data = img_file.read()
        
        # Extract feature vector
        feature_vector = extract_features(img_data)
        
        # Find similar items
        limit = int(request.form.get('limit', 5))
        similar_items = mongodb.find_similar_items(feature_vector, limit)
        
        # Convert ObjectId to string for JSON serialization
        for item in similar_items:
            item['_id'] = str(item['_id'])
        
        return jsonify({
            "success": True,
            "items": similar_items
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/get-item/<item_id>', methods=['GET'])
def api_get_item(item_id):
    """API endpoint to get an item by ID"""
    try:
        item = mongodb.find_item_by_id(item_id)
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
        
        # Convert ObjectId to string for JSON serialization
        item['_id'] = str(item['_id'])
        
        # Remove feature vector from response (optional)
        if 'feature_vector' in item:
            del item['feature_vector']
        
        return jsonify({
            "success": True,
            "item": item
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)