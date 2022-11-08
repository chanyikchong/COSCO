import requests
import json
from string import Template
import framework.server.common.codes as codes


class InvalidResponse:
    text = {'server_error': ''}


REQUEST_TEMPLATE = Template("http://$HOST:$PORT/request")
CONTAINER_PORT = 8081


def handle_request(payload, host, framework):
    client_url = REQUEST_TEMPLATE.substitute(HOST=host, PORT=CONTAINER_PORT)
    print(json.dumps(payload), client_url)
    try:
        resp = requests.get(client_url, data=json.dumps(payload), timeout=360)
    except Exception as e:
        resp = InvalidResponse()
        resp.text = json.dumps({'server_error': str(e) + ' for payload = ' + json.dumps(payload)})
    framework.logger.debug("Response received by server from agent " + host + " : " + resp.text)
    return json.loads(resp.text)
