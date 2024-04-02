# ----------------------------------------------------
# Copyright (c) 2023. 挥杯劝, Inc. All Rights Reserved
# @作者         : 挥杯劝(Huibq)
# @邮件         : ***
# [url=home.php?mod=space&uid=81836]@文件[/url]         : 小胖VPN.py
# @创建时间     : 2023/12/24 13:40
# ---------------------------------------------------
import json
import time
import uuid
import requests
import urllib3
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Cipher import AES
from Crypto.Hash import MD5
from Crypto.Util.Padding import unpad
import base64

urllib3.disable_warnings()
proxies = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809",
}


def invite():
    url = 'https://***.fatvpn.pro/api/addRefereeToUserReferral/'
    headers = {
        'accept': 'application/json',
        'accept-charset': 'UTF-8',
        'cache-control': 'max-age=1800',
        'user-agent': 'Ktor client',
        'content-type': 'application/x-www-form-urlencoded',
        'content-length': '0',
        'accept-encoding': 'gzip'

    }
    number = 0
    while number < 10:
        Id = uuid.uuid4()
        data = {
            'uniqueId': Id,
            'referralCode': ''
        }
        req = requests.post(url, data=data, headers=headers, proxies=proxies, verify=False)
        print(req.text)
        number += 1
        time.sleep(3)


def decrypt_rsa(data):
    private_key = '''MIIEogIBAAKCAQB/xKmjpGbc6MdYYp0v32JLpIaOvyTrhupgWWcQfRFyuk51p3fof6Eh+M3qp00yQzHtP/MkRP6Ldc5DOP6D9b2s522s7vpzsG3LtBenZZ8xN4usMBuOTQzJTvLfCjfLl0gXBDJneKjHPKxLxaPmsFwTS1Mi3cZEMNoA8ns9hmdDwcsCLBAHzTjpNcfQN+EQw7mRbKK4n14lLzSfYjjScuE/Oj8WOpy5Wl0/UAoHRLbujPNU26hJlpQa2S6ipvFIBZHjhaU4AesST55XBOLDmPNCaKQV9Nf9RJ2GABazioPAOo+Q7iEfEXysNcuhdP7q8gXJ0oNHdyJl6PAZZddiZyB7AgMBAAECggEASJMC8Orvasfmg7PwKUMv6FuZ+vdkF0zZUMU3n8wK3yooavgnSi9E7bEP9hv143j7oRHUIGP4WmseMFztZTNu/Amw6KwOIyyyESVI0lMM673rXnEtFdV6T9bCaiK5srFJx5kgsFl/NTyneZrYEK9YfbUpkgJ7HjzJeAREMJxph7h7FsSd2M+MB7m/J+6TGZ6fIsZFcy0vIQLjLSAIOuO5T1HEB+m7AFHyvXdAfR+BWKSTIs0pkC2wUgbE6CsMlzx3XvM1DnmUiDF478t0UFKEnpQKsnmRSweH7htbmhigiP9kHcidJ8+PNqojy+DGsCjuIoSiOqgGEs1+woYSw34pIQKBgQDPmwiPNYxM2OYP7onoBhsBKDmdxGSUnEFmUgmtPpEy5PssJDz/SEoVponyW/dPEycn8Bjc+6fnjUi0RuqE2e3JnpSWrhhLBEFhrd16bHVwaHA40IWM8UYsgUjmTPdt2DO9+M8ZW+UADUAAlARm/NSZo+Z4PzTBIjW4e4xhxm/2AwKBgQCdjUs4VC0qdMHT5ntE5IWIagrXxshacaE8rTAgjYOMaib0Skq593yqksb1kiIF3CeWZACdd6qk6Gah5Oc755K8wuixxXZY8vzwzLf5K3NqFkbAhxR251kZoQvmoP/OJDbBxO0Y86auuX7m498vIh7raoVWjzXKSHK2/ZyTfEw+KQKBgFH5r7mMtWeqxb1IvZ+muYcNcSLA585enNxgTH3iFMd570wQyx0qWEaQSiwu8EqDD5UPk2G+5R/jg+/biMMIooJYYefVurX0ajS9yJSMuxq1wopMnE94/fKY4kY94f23v0amNnCW/qe0k68mw04/S1uXgmu82YHhlkDQWDBLgO4tAoGAS6xv8rBLuVa3OoY7sw1oLetxJc7usLJfVXuB4EDYbHsYFsIQPl5m3K7/LThxawshYJTLztaJege+NAh0IEvMKSodBjXn8DVV1Hsf6mg6WTw144d+BtZ771lxE+dEtsiiHFPv5coxxz6Fe3T73/GtlDlnrfm/Rleh8c7Cg/xxynECgYEAkIj8otP2kFIqR+5tTl2u3wINlVyPONyWKT0GNWN9am80sUMpYYvI5TUmJ47+2709GgiYQWZxSsNCqVPuKM57u1czOo7iRqD2UavL3GiN4zmBbGQ+4TqwPJb1k2Y8dpIMwIGRKT9GWLexXYqBLK1JJV3TrJ4CjEA9A/Txy/HByZg='''
    key = RSA.importKey(base64.urlsafe_b64decode(private_key))
    cipher = PKCS1_OAEP.new(key)
    decrypted_message = cipher.decrypt(base64.b64decode(data))
    return decrypted_message.decode()


def decrypt_aes(key, data):
    key = MD5.new(key.encode()).digest()
    iv = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(base64.b64decode(data)), AES.block_size)
    return decrypted_data.decode()


def get_node():
    url = 'http://***.fatvpn.pro/api/v2/vpnServerNodes?version=84'
    header = {
        'authorization': 'N0Q6Q0Y6Nzg6RDE6',
        'cache-control': 'no-cache',
        'accept': 'application/json',
        'accept-charset': 'UTF-8',
        'user-agent': 'Ktor client',
        'content-type': 'text/plain;charset=UTF-8',
        'content-length': '0',
        'accept-encoding': 'gzip'
    }
    text = 'com.fat.vpn0OO0UGxlYXNlIHVzZSB0aGUgcmlnaHQgYXBw'
    req = requests.post(url, data=text, headers=header, verify=False).json()
    key = req['key']
    key = decrypt_rsa(key)
    node_info = decrypt_aes(key, req['data'])
    for server_list in json.loads(node_info):
        if server_list['servers']:
            for servers in server_list['servers']:
                if servers:
                    server = servers['server']
                    ip = servers['ip']
                    if ip:
                        address = '小胖'
                        a = server.split('@')
                        b = a[1].split(':')[1].split('#')
                        vless = a[0] + '@' + ip + ':' + b[0] + '#' + address
                        print(vless)
                    else:
                        print(server)


get_node()