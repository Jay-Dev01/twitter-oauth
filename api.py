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
CALLBACK_URL = "https://ubiquitous-lolly-8d1bc5.netlify.app/"
TWITTER_CLIENT_SECRET = "go5Cl9Us7eCdr6PKzb7GeFX-IppV-gY9iI3RBWc7x7GNtE93PV"
twitter_tokens = {}  
twitter_connection_state = {}  # Track which agent is being connected: { wallet: agent_id }

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
        self.contract_address = ""  # Initialize as empty string

# Change the agents storage to be wallet-specific
# Instead of a simple dict, we'll use a nested dict where the first key is the wallet address
agents = {}  # Format: { wallet_address: { agent_id: Agent } }

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
    """Handle both Twitter OAuth initialization and callback"""
    try:
        data = request.json
        agent_id = data.get('agent_id')
        oauth_token = data.get('oauth_token')
        oauth_verifier = data.get('oauth_verifier')
        
        # If we have oauth_token and oauth_verifier, this is a callback
        if oauth_token and oauth_verifier:
            print(f"Processing callback for wallet: {current_wallet}")
            print(f"Current connection state: {twitter_connection_state}")
            print(f"Current tokens: {twitter_tokens}")
            
            # Get the agent_id from stored state
            agent_id = twitter_connection_state.get(current_wallet)
            if not agent_id:
                return jsonify({"error": "No pending Twitter connection"}), 400

            # Verify agent exists
            wallet_agents = agents.get(current_wallet, {})
            agent = wallet_agents.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404

            try:
                # Complete OAuth flow
                oauth = OAuth1Session(
                    client_key=API_KEY,
                    client_secret=API_SECRET,
                    resource_owner_key=twitter_tokens.get(f"{current_wallet}_request_token"),
                    resource_owner_secret=twitter_tokens.get(f"{current_wallet}_request_secret"),
                    verifier=oauth_verifier
                )

                # Get the access token
                oauth_tokens = oauth.fetch_access_token('https://api.twitter.com/oauth/access_token')
                
                # Store the access tokens
                twitter_tokens[f"{current_wallet}_oauth1_token"] = oauth_tokens.get('oauth_token')
                twitter_tokens[f"{current_wallet}_oauth1_secret"] = oauth_tokens.get('oauth_token_secret')

                # Initialize Tweepy client
                client = tweepy.Client(
                    consumer_key=API_KEY,
                    consumer_secret=API_SECRET,
                    access_token=oauth_tokens.get('oauth_token'),
                    access_token_secret=oauth_tokens.get('oauth_token_secret')
                )
                
                # Get user info
                user = client.get_me(user_fields=['username', 'name', 'id'])
                if not user or not user.data:
                    raise Exception("Failed to get user data")
                
                twitter_user = {
                    'username': user.data.username,
                    'name': user.data.name,
                    'id': str(user.data.id)
                }

                # Update agent's connected_twitter status
                agent.connected_twitter = user.data.username
                print(f"Agent connected to twitter: {agent.connected_twitter}")
                # Clean up connection state
                del twitter_connection_state[current_wallet]
                
                return jsonify({
                    "success": True,
                    "twitter_user": twitter_user,
                    "agent": agent.__dict__
                }), 200

            except Exception as e:
                print(f"OAuth completion error: {str(e)}")
                return jsonify({"error": f"OAuth completion failed: {str(e)}"}), 500

        # If no oauth tokens, this is initial auth request
        else:
            print(f"Starting Twitter auth for wallet: {current_wallet}, agent: {agent_id}")
            
            # Verify agent exists
            wallet_agents = agents.get(current_wallet, {})
            if agent_id not in wallet_agents:
                return jsonify({"error": "Agent not found"}), 404

            # Store which agent is being connected
            twitter_connection_state[current_wallet] = agent_id
            
            # Initialize OAuth session
            oauth = OAuth1Session(
                client_key=API_KEY,
                client_secret=API_SECRET,
                callback_uri=CALLBACK_URL
            )

            try:
                response = oauth.fetch_request_token('https://api.twitter.com/oauth/request_token')
                
                # Store request tokens
                twitter_tokens[f"{current_wallet}_request_token"] = response.get('oauth_token')
                twitter_tokens[f"{current_wallet}_request_secret"] = response.get('oauth_token_secret')
                print(f"Stored tokens: {twitter_tokens}")
                # Get user info
                
                client1 = tweepy.Client(
                consumer_key=API_KEY,
                consumer_secret=API_SECRET,
                access_token=response.get('oauth_token'),
                access_token_secret=response.get('oauth_token_secret')
                )
                user = client1.get_me(user_fields=['username', 'name', 'id'])
                if not user or not user.data:
                    raise Exception("Failed to get user data")
                
                twitter_user = {
                    'username': user.data.username,
                    'name': user.data.name,
                    'id': str(user.data.id)
                }

                # Update agent's connected_twitter status
                agent.connected_twitter = user.data.username
                print(f"Agent connected to twitter: {agent.connected_twitter}")
                
                # Create authorization URL
                auth_url = f"https://api.twitter.com/oauth/authorize?oauth_token={response.get('oauth_token')}"
                
                return jsonify({"auth_url": auth_url}), 200
                
            except Exception as e:
                print(f"Request token error: {str(e)}")
                return jsonify({"error": "Failed to get request token"}), 500
            
    except Exception as e:
        print(f"Twitter auth error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/twitter/status/<agent_id>', methods=['GET'])
@token_required
def check_twitter_status(current_wallet, agent_id):
    """Check if a specific agent has Twitter connected"""
    try:
        print(f"Checking status for wallet: {current_wallet}, agent: {agent_id}")
        
        # Check if agent exists and belongs to wallet
        wallet_agents = agents.get(current_wallet, {})
        agent = wallet_agents.get(agent_id)
        
        if not agent:
            print("Agent not found")
            return jsonify({
                "connected": False,
                "error": "Agent not found"
            }), 404

        # Get OAuth tokens using consistent naming
        oauth1_token_key = f"{current_wallet}_oauth1_token"
        oauth1_secret_key = f"{current_wallet}_oauth1_secret"

        # If we have tokens and a connected username, verify the connection
        if (oauth1_token_key in twitter_tokens and 
            oauth1_secret_key in twitter_tokens and 
            agent.connected_twitter):
            
            try:
                client = tweepy.Client(
                    consumer_key=API_KEY,
                    consumer_secret=API_SECRET,
                    access_token=twitter_tokens[oauth1_token_key],
                    access_token_secret=twitter_tokens[oauth1_secret_key]
                )
                
                user = client.get_me(user_fields=['username', 'name'])
                
                if user.data and user.data.username == agent.connected_twitter:
                    return jsonify({
                        "connected": True,
                        "twitter_user": {
                            "username": user.data.username,
                            "name": user.data.name,
                            "id": str(user.data.id)
                        }
                    }), 200
                
            except Exception as e:
                print(f"Twitter API error: {str(e)}")
                agent.connected_twitter = False

        # If we get here, either there's no connection or verification failed
        return jsonify({
            "connected": False,
            "error": "Twitter not connected"
        }), 200  # Changed to 200 to avoid 404
            
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
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
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
    # Return only agents belonging to the current wallet
    wallet_agents = agents.get(current_wallet, {})
    return jsonify([agent.__dict__ for agent in wallet_agents.values()])

@app.route('/api/agents/<agent_id>', methods=['GET'])
@token_required
def get_agent(current_wallet, agent_id):
    # Check if wallet has any agents
    wallet_agents = agents.get(current_wallet, {})
    agent = wallet_agents.get(agent_id)
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
        description=data.get('description', ''),
        goal=data.get('goal', ''),
        functions=[],
        connected_twitter=False
    )
    
    # Initialize wallet's agents dict if it doesn't exist
    if current_wallet not in agents:
        agents[current_wallet] = {}
    
    agents[current_wallet][agent_id] = agent
    return jsonify(agent.__dict__), 201

@app.route('/api/agents/<agent_id>', methods=['PATCH'])
@token_required
def patch_agent(current_wallet, agent_id):
    # Check if wallet has any agents
    wallet_agents = agents.get(current_wallet, {})
    if agent_id not in wallet_agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    data = request.json
    agent = wallet_agents[agent_id]
    
    # Update only the fields that are provided
    if 'name' in data:
        agent.name = data['name']
    if 'symbol' in data:
        agent.symbol = data['symbol']
    if 'description' in data:
        agent.description = data['description']
    if 'goal' in data:
        agent.goal = data['goal']
    if 'contract_address' in data:
        agent.contract_address = data['contract_address']
    if 'functions' in data:
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
    # Check if wallet has any agents
    wallet_agents = agents.get(current_wallet, {})
    if agent_id not in wallet_agents:
        return jsonify({'error': 'Agent not found'}), 404
    
    del wallet_agents[agent_id]
    # Remove wallet entry if no agents left
    if not wallet_agents:
        del agents[current_wallet]
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
