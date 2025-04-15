# # add_info.py

# import base64
# import gridfs
# import numpy as np
# from pymongo import MongoClient
# from bson import ObjectId
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # MongoDB connection
# client = MongoClient(os.getenv("MONGODB_URI"))
# db = client[os.getenv("MONGO_DB_NAME")]
# fs = gridfs.GridFS(db)
# collection = db[os.getenv("MONGO_COLLECTION_NAME")]

# def save_to_mongodb(image_file, feature_vector, item):
#     """
#     Stores image, metadata, and feature vector in MongoDB
#     """
#     # Save image to GridFS
#     image_id = fs.put(image_file, filename=item.get('id', 'unknown'))

#     # Convert feature vector to list (for BSON compatibility)
#     feature_vector = feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else feature_vector

#     doc = {
#         "id": item.get('id', ''),
#         "gender": item.get('gender', ''),
#         "age": item.get('age', ''),
#         "type": item.get('type', ''),
#         "description": item.get('description', ''),
#         "feature_vector": feature_vector,
#         "image_file_id": image_id
#     }

#     inserted = collection.insert_one(doc)
#     return str(inserted.inserted_id)

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId
from extract_feature import extract_features

# Load environment variables
load_dotenv()

class MongoDB:
    def __init__(self):
        # Get MongoDB URI from environment variables
        self.mongo_uri = os.getenv("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MongoDB URI not found in environment variables")
        
        # Get database name (or use default)
        self.db_name = os.getenv("MONGODB_DB_NAME", "mydatabase")
        
        # Create MongoDB client and connect to database
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]  # Use bracket notation to specify database
        self.collection = self.db.items

    def add_item(self, data):
        """
        Add item details to MongoDB with feature vector
        
        Parameters:
        - data: dictionary containing item details (id, gender, age, type, description, img)
        
        Returns:
        - Inserted document ID
        """
        try:
            # Extract required fields
            item_id = data.get('id', '')
            gender = data.get('gender', '')
            age = data.get('age', '')
            item_type = data.get('type', '')
            description = data.get('description', '')
            img_data = data.get('img')
            
            # Extract feature vector from image
            feature_vector = extract_features(img_data)
            
            # Prepare document for MongoDB
            document = {
                "item_id": item_id,
                "gender": gender,
                "age": age,
                "type": item_type,
                "description": description,
                "feature_vector": feature_vector,
                "created_at": datetime.now()
            }
            
            # Insert document
            result = self.collection.insert_one(document)
            
            return str(result.inserted_id)
        
        except Exception as e:
            print(f"Error adding item to MongoDB: {e}")
            raise
    
    def find_item_by_id(self, item_id):
        """Find an item by its ID"""
        return self.collection.find_one({"item_id": item_id})
    
    def find_similar_items(self, feature_vector, limit=5):
        """
        Find similar items based on feature vector using MongoDB's $nearSphere
        
        Note: This requires a 2dsphere index on feature_vector field
        """
        # Create index if it doesn't exist
        self.collection.create_index([("feature_vector", "2dsphere")])
        
        pipeline = [
            {
                "$search": {
                    "knnBeta": {
                        "vector": feature_vector,
                        "path": "feature_vector",
                        "k": limit
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "item_id": 1,
                    "gender": 1,
                    "age": 1,
                    "type": 1,
                    "description": 1,
                    "created_at": 1,
                    "score": {
                        "$meta": "searchScore"
                    }
                }
            }
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()

# Initialize MongoDB singleton
mongodb = MongoDB()

def add_info(data):
    """
    Function to add information to MongoDB.
    This is the main entry point for the API.
    """
    return mongodb.add_item(data)

# Test function
if __name__ == "__main__":
    # Example usage
    test_data = {
        "id": "test123",
        "gender": "male",
        "age": "adult",
        "type": "shirt",
        "description": "Blue cotton shirt",
        "img": "sample_img/test.jpg"  # Path to test image
    }
    
    inserted_id = add_info(test_data)
    print(f"Document inserted with ID: {inserted_id}")