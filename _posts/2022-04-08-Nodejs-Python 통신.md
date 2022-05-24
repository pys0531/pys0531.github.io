---
title:  <font size="5">Nodejs-Python 통신</font>
excerpt: "Nodejs-Python 통신"
toc: true
toc_sticky: true
categories:
  - Server
tags:
  - Server
  - Python
  - Nodejs
last_modified_at: 2022-04-08T22:39:00-55:00
---

<font size="3">




</font> 

**Python 코드 (서버)**

```python
import socket 
from _thread import *


# 쓰레드에서 실행되는 코드입니다. 
# 접속한 클라이언트마다 새로운 쓰레드가 생성되어 통신을 하게 됩니다. 
def threaded(client_socket, addr): 
    print('Connected by :', addr[0], ':', addr[1]) 

    # 클라이언트가 접속을 끊을 때 까지 반복합니다. 
    while True: 
        try:
            # 데이터가 수신되면 클라이언트에 다시 전송합니다.(에코)
            data = client_socket.recv(1024)

            if not data: 
                print('A Disconnected by ' + addr[0],':',addr[1])
                break

            print('Received from ' + addr[0],':',addr[1] , data.decode())
            client_socket.send(data) 

        except ConnectionResetError as e:
            print('B Disconnected by ' + addr[0],':',addr[1])
            break
             
    client_socket.close() 


HOST = '127.0.0.1'
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT)) 
server_socket.listen(40) 

print('server start')


# 클라이언트가 접속하면 accept 함수에서 새로운 소켓을 리턴합니다.
# 새로운 쓰레드에서 해당 소켓을 사용하여 통신을 하게 됩니다. 
while True: 
    print('wait')
    client_socket, addr = server_socket.accept() 
    start_new_thread(threaded, (client_socket, addr)) 

server_socket.close()
```

<br>
**NodeJS 코드 (클라이언트)**

```js
var request = require('request');

var ioOut = require('socket.io-client');
var socketOut = ioOut.connect('http://localhost:8881'); // 8881로 보낸다. 응답은

var msg2 = 'Test'

// 소켓서버
var ioServerSocket = require('socket.io').listen(8881);
// 누군가 Socket 으로 던졌을때 받는다.
ioServerSocket.on('connection', function(socketIn) {
  socketIn.on('idle', function(msg) {
    //socketOut.emit('bar', msg);
    console.log('Client Socket dummy!+msg='+msg)


    // Start to python --->
    client.connect(9999, '127.0.0.1', function() {
      console.log('Connected222');
      client.write('Hello222, server! Love, Client.');
      //client.write('');
    });
    // End to python --->
    
  });
  socketIn.on('busy', function(msg) {
    //socketOut.emit('bar', msg);
    console.log('Client Socket BUSY='+msg)
  });
});


//------------ Start of CLIENT --
var net = require('net');

var client = new net.Socket();
client.connect(9999, '127.0.0.1', function() {
  console.log('Connected');
  client.write('Hello, server! Love, Client.');
});


client.on('data', function(data) {
  console.log('Received from PY: ' + data);
  client.destroy(); // kill client after server's response
});


client.on('close', function() {
  console.log('Connection closed');
});
// ------------ End of CLIENT --
// Test
var http = require('http');
var i = 1;
function onReq(request, response) {
    console.log('request received2 ='+i);
    i++;
    response.writeHead(200, {'Content-Type':'text/plain'})
    response.write('Hello Wo');
    response.end();

    socketOut.emit('idle', 'dummy');
}
http.createServer(onReq).listen(8888);
console.log('test msg2='+msg2)
console.log('Server Start!')
```
