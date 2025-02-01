import os
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
import jwt as pyjwt
import base64
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from dotenv import load_dotenv
import requests
from requests_oauthlib import OAuth1Session
import tweepy
import json
from pathlib import Path
from solders.keypair import Keypair
import uuid

app = Flask(__name__)
CORS(app, supports_credentials=True)
load_dotenv()

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['API_KEY'] = os.environ.get('API_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = '/opt/render/project/src/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

data_store = {}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

API_KEY="9IyUfQlzjWrcvEzoQRJJGgPnq"
API_SECRET="kadR7zIRjzLfsw0bb19b5GwYt9xw4LXeLht6QreeNxUPAUE2Kn"
TWITTER_CLIENT_ID = "V09mVVg2QVFoM0d4Q3JmM09Gd086MTpjaQ"
CALLBACK_URL = "https://caring-follow-415683.framer.app/"
TWITTER_CLIENT_SECRET = "go5Cl9Us7eCdr6PKzb7GeFX-IppV-gY9iI3RBWc7x7GNtE93PV"
twitter_tokens = {}  

# Add this model class
class Agent:
    def __init__(self, id, name, symbol, description, goal, functions, connected_twitter):
        self.id = id
        self.name = name
        self.symbol = symbol
        self.description = description
        self.goal = goal
        self.functions = functions
        self.connected_twitter = connected_twitter

# In-memory storage (replace with database in production)
agents = {}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Authorization header is missing'}), 401
        
        try:
            parts = auth_header.split()
            if parts[0].lower() != 'bearer' or len(parts) != 2:
                return jsonify({'error': 'Invalid token format'}), 401
                
            token = parts[1]
            
            data = pyjwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_wallet = data['wallet_id']
            
        except pyjwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except pyjwt.InvalidTokenError as e:
            return jsonify({'error': f'Invalid token: {str(e)}'}), 401
        except Exception as e:
            print(f"Token validation error: {str(e)}")  
            return jsonify({'error': f'Token validation error: {str(e)}'}), 401
            
        return f(current_wallet, *args, **kwargs)
    
    return decorated


@app.route('/api/twitter/auth', methods=['POST'])
@token_required
def twitter_auth(current_wallet):
    """Start Twitter OAuth process for a wallet"""
    
    try:
        # Initialize OAuth 1.0a session
        oauth = OAuth1Session(
            client_key=API_KEY,  # Api key
            client_secret=API_SECRET,  # Api secret
            callback_uri=CALLBACK_URL # For PIN-based auth 'oob'
        )

        # Get OAuth 1.0a request token
        try:
            response = oauth.fetch_request_token('https://api.twitter.com/oauth/request_token')
            print(response)
            # Store request tokens for later use
            twitter_tokens[f"{current_wallet}_request_token"] = response.get('oauth_token')
            twitter_tokens[f"{current_wallet}_request_secret"] = response.get('oauth_token_secret')
            
            # Create authorization URL
            auth_url = f"https://api.twitter.com/oauth/authorize?oauth_token={response.get('oauth_token')}"
            
            print(f"Generated OAuth 1.0a URL: {auth_url}")
            return jsonify({"auth_url": auth_url}), 200
            
        except Exception as e:
            print(f"Request token error: {str(e)}")
            return jsonify({"error": "Failed to get request token"}), 500
            
    except Exception as e:
        print(f"Twitter auth error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@token_required
def check_twitter_status(current_wallet):
    """Check if a wallet has Twitter connected"""
    try:
        # Check if wallet has a Twitter token stored
        if current_wallet in twitter_tokens:
            # Test the token by making a simple Twitter API call
            headers = {
                'Authorization': f'Bearer {twitter_tokens[current_wallet]}'
            }
            
            # Call Twitter's API to verify the token
            response = requests.get(
                'https://api.twitter.com/2/users/me',
                headers=headers
            )
            
            if response.ok:
                user_data = response.json()
                return jsonify({
                    "connected": True,
                    "twitter_user": user_data['data']
                }), 200
            else:
                # Token might be invalid
                twitter_tokens.pop(current_wallet, None)  # Remove invalid token
                return jsonify({
                    "connected": False,
                    "error": "Invalid Twitter token"
                }), 401
        
        return jsonify({
            "connected": False,
            "error": "No Twitter connection found"
        }), 404
        
    except Exception as e:
        print(f"Status check error: {str(e)}")
        return jsonify({
            "connected": False,
            "error": str(e)
        }), 500
    

@app.route('/api/twitter/post', methods=['POST'])
@token_required
def post_to_twitter(current_wallet):
    """Post the wallet's tweet to Twitter"""
    try:
        oauth1_token_key = f"{current_wallet}_oauth1_token"
        oauth1_secret_key = f"{current_wallet}_oauth1_secret"
        
        if oauth1_token_key not in twitter_tokens or oauth1_secret_key not in twitter_tokens:
            return jsonify({"error": "Not authorized with Twitter"}), 401

        try:
            # Retrieve the access token and secret for the current wallet
            access_token = twitter_tokens[f"{current_wallet}_oauth1_token"]
            access_token_secret = twitter_tokens[f"{current_wallet}_oauth1_secret"]
            
            # Create v2 connection for tweeting
            client_v2 = tweepy.Client(
                consumer_key=API_KEY,   #Api key
                consumer_secret=API_SECRET,  #Api secret
                access_token=access_token,
                access_token_secret=access_token_secret
            )

            # Get the tweet text from data store
            tweet_text = data_store.get(current_wallet, {}).get("tweet", "")
            if not tweet_text:
                return jsonify({"error": "No tweet text found for this wallet"}), 404

            # Create tweet using v2
            response = client_v2.create_tweet(text=tweet_text)

            print(f"Tweet created: {response}")
            
            if response.data:
                tweet_id = response.data['id']
                return jsonify({
                    "message": "Successfully posted to Twitter",
                    "tweet_url": f"https://twitter.com/user/status/{tweet_id}"
                }), 200
            else:
                return jsonify({"error": f"Tweet creation failed: {response}"}), 500
                
        except Exception as e:
            print(f"Twitter API error: {str(e)}")
            return jsonify({"error": f"Twitter API error: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error posting to Twitter: {str(e)}")
        return jsonify({"error": str(e)}), 500


def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != app.config["API_KEY"]:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-API-Key')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/auth', methods=['POST'])
def authenticate():
    try:
        print("Received auth request")  
        
        data = request.get_json()
        print("Request data:", data)  
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        required_fields = ['wallet_id', 'message', 'signature']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
        wallet_id = data['wallet_id']
        message = data['message']
        signature = data['signature']
        
        try:
            signature_bytes = base64.b64decode(signature)
            print("Signature decoded successfully")  
            
            token = pyjwt.encode({
                'wallet_id': wallet_id,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            
            print("Token generated successfully:", token)  
            
            return jsonify({'token': token}), 200
            
        except Exception as e:
            print(f"Error processing signature: {str(e)}")  
            return jsonify({'error': f'Invalid signature format: {str(e)}'}), 401
        
    except Exception as e:
        print(f"Authentication error: {str(e)}")  
        return jsonify({'error': str(e)}), 500


@app.route('/api/data', methods=['POST'])
@token_required
def post_data(current_wallet):
    payload = request.get_json()
    if not payload or 'tweet' not in payload or 'needNewTweet' not in payload:
        return jsonify({"error": "Missing required fields in request body"}), 400

    tweet = payload['tweet']
    need_new_tweet = payload['needNewTweet']
    
    if current_wallet not in data_store:
        data_store[current_wallet] = {}
    
    data_store[current_wallet]["tweet"] = tweet
    data_store[current_wallet]["needNewTweet"] = need_new_tweet
    
    return jsonify({
        "message": "Data received successfully", 
        "walletId": current_wallet,
        "data": data_store[current_wallet]
    }), 200

@app.route('/api/data', methods=['DELETE'])
@api_key_required
def delete_all_data():
    global data_store
    data_store = {}
    return jsonify({"message": "All data cleared successfully!"}), 200
    
    
@app.route('/api/prepare-token', methods=['POST'])
@token_required
def prepare_token_endpoint(current_wallet):
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Get token details from form data
        name = request.form.get('name', '')
        symbol = request.form.get('symbol', '')
        description = request.form.get('description', '')
        twitter = request.form.get('twitter', '')
        telegram = request.form.get('telegram', '')
        website = request.form.get('website', '')
        
        # Validate required fields
        if not all([name, symbol]):
            return jsonify({"error": "Name and symbol are required"}), 400
            
        # Generate keypair for token
        mint_keypair = Keypair()
        
        # Save file temporarily
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{current_wallet}_{file.filename}")
        file.save(temp_file_path)
        
        try:
            # Prepare form data for IPFS upload
            form_data = {
                'name': name,
                'symbol': symbol,
                'description': description or 'Token created via PumpPortal.fun',
                'twitter': twitter,
                'telegram': telegram,
                'website': website,
                'showName': 'true'
            }
            
            # Upload file to IPFS
            with open(temp_file_path, 'rb') as f:
                files = {'file': (file.filename, f, 'image/png')}
                metadata_response = requests.post(
                    "https://pump.fun/api/ipfs",
                    data=form_data,
                    files=files
                )
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            if metadata_response.status_code != 200:
                raise Exception(f"IPFS upload failed: {metadata_response.status_code} {metadata_response.reason}")
                
            metadata_response_json = metadata_response.json()
            
            # Get secret key bytes
            secret_bytes = bytes(mint_keypair)
            
            # Create preparation info
            prep_info = {
                "mintPublicKey": str(mint_keypair.pubkey()),
                "mintSecretKey": base64.b64encode(secret_bytes).decode('utf-8'),
                "metadata": {
                    "name": metadata_response_json['metadata']['name'],
                    "symbol": metadata_response_json['metadata']['symbol'],
                    "uri": metadata_response_json['metadataUri']
                }
            }
            
            # Store prep info in data store for this wallet
            data_store[current_wallet] = {
                **data_store.get(current_wallet, {}),
                "token_prep": prep_info
            }
            
            return jsonify({
                "success": True,
                "data": prep_info
            }), 200
            
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise e
            
    except Exception as error:
        print(f"Error preparing token: {str(error)}")
        return jsonify({
            "error": str(error),
            "details": getattr(error, 'response', {}).get('text', '')
        }), 500
    
@app.route('/api/upload', methods=['POST'])
@token_required
def upload_image(current_wallet):
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Change this line to match the expected filename format
    filename = f"image_{current_wallet}.png"  # Remove the random hex string
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    print(f"Uploading file to: {filepath}")  # Debugging log

    try:
        file.save(filepath)
        print(f"File saved successfully at: {filepath}")  # Confirm file saved
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"error": "File save failed"}), 500

    return jsonify({"message": "File uploaded successfully", "filePath": f"/api/images/{filename}"}), 200
    

# Add these new routes
@app.route('/api/agents', methods=['GET'])
@token_required
def get_agents(current_wallet):
    return jsonify(list(agents.values()))

@app.route('/api/agents/<agent_id>', methods=['GET'])
@token_required
def get_agent(current_wallet, agent_id):
    agent = agents.get(agent_id)
    if agent is None:
        return jsonify({'error': 'Agent not found'}), 404
    return jsonify(agent.__dict__)

@app.route('/api/agents', methods=['POST'])
@token_required
def create_agent(current_wallet):
    data = request.json
    
    # Only require name and symbol for initial creation
    if not data.get('name') or not data.get('symbol'):
        return jsonify({'error': 'Name and symbol are required'}), 400

    agent_id = str(uuid.uuid4())
    
    agent = Agent(
        id=agent_id,
        name=data['name'],
        symbol=data['symbol'],
        description=data.get('description', ''),  # Optional fields with defaults
        goal=data.get('goal', ''),
        functions=[],  # Start with empty functions
        connected_twitter=False  # Start disconnected
    )
    
    agents[agent_id] = agent
    return jsonify(agent.__dict__), 201

@app.route('/api/agents/<agent_id>', methods=['PATCH'])
@token_required
def patch_agent(current_wallet, agent_id):
    if agent_id not in agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    data = request.json
    agent = agents[agent_id]
    
    # Update only the fields that are provided
    if 'name' in data:
        agent.name = data['name']
    if 'symbol' in data:
        agent.symbol = data['symbol']
    if 'description' in data:
        agent.description = data['description']
    if 'goal' in data:
        agent.goal = data['goal']
    if 'functions' in data:
        # Validate functions if they're being updated
        valid_functions = {
            'post_tweet': {'service': 'Twitter', 'description': 'Create and publish a new tweet'},
            'reply_tweet': {'service': 'Twitter', 'description': 'Reply to an existing tweet'},
            'like_tweet': {'service': 'Twitter', 'description': 'Like a specific tweet'}
        }
        
        for func in data['functions']:
            if func not in valid_functions:
                return jsonify({'error': f'Invalid function: {func}'}), 400
                
        agent.functions = data['functions']
        
    if 'connected_twitter' in data:
        agent.connected_twitter = data['connected_twitter']
    
    return jsonify(agent.__dict__)

@app.route('/api/agents/<agent_id>', methods=['DELETE'])
@token_required
def delete_agent(current_wallet, agent_id):
    if agent_id not in agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    del agents[agent_id]
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
