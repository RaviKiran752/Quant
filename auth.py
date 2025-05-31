import urllib.parse
import requests
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Extract the authorization code from the callback URL
        query_components = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if 'code' in query_components:
            self.server.auth_code = query_components['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authorization successful! You can close this window.")
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authorization failed! Please try again.")

def get_upstox_access_token(api_key, api_secret, redirect_url):
    """
    Get access token from Upstox API using OAuth2 flow
    """
    # Start a local server to receive the callback
    server = HTTPServer(('localhost', 5000), CallbackHandler)
    server.auth_code = None
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Generate and open login URL
    login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_url}&state=random_state"
    print(f"Opening login URL in your browser...")
    webbrowser.open(login_url)
    
    # Wait for the callback
    while server.auth_code is None:
        pass
    
    # Stop the server
    server.shutdown()
    server.server_close()
    
    # Exchange authorization code for access token
    token_url = "https://api.upstox.com/v2/login/authorization/token"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Api-Version": "2.0"
    }
    data = {
        "code": server.auth_code,
        "client_id": api_key,
        "client_secret": api_secret,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_url,
        "state": "random_state"
    }
    
    response = requests.post(token_url, headers=headers, data=data)
    
    if response.status_code == 200:
        access_token = response.json()["access_token"]
        print("Successfully obtained access token!")
        return access_token
    else:
        print(f"Error getting access token: {response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    api_key = '153bedca-f4ed-4f13-9cc3-5723a16333bc'
    api_secret = 'z9p19ve2ck'
    redirect_url = 'http://localhost:5000/callback'
    
    access_token = get_upstox_access_token(api_key, api_secret, redirect_url)
    if access_token:
        print("Access token:", access_token)
        # Save the token to a file for future use
        with open('access_token.json', 'w') as f:
            json.dump({'access_token': access_token}, f) 