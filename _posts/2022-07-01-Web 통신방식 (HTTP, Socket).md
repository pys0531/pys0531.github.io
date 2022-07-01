---
title:  <font size="5">Web 통신방식 (HTTP, Socket)</font>
excerpt: "Web 통신방식 (HTTP, Socket)"
toc: true
toc_sticky: true
categories:
  - Server
tags:
  - Server
  - Socket
  - HTTP
  - Python
  - Nodejs
last_modified_at: 2022-04-08T22:39:00-55:00
---


<font size="3">
<div markdown = "1">
일반적으로 클라이언트에서 서버로 부터 데이터를 가져오기 위한 통신은 HTTP통신과 Socket통신이 있다.
HTTP와 Socket 통신의 차이점을 간단히 적어본다.
 
## 1. HTTP(HyperText Transfer Protocol) 통신
  - HTML 파일을 전송하려는 목적으로 만들어졌으나 현재는 Json, Image 파일 등도 전송
  - 웹을 통한 통신
  - 클라이언트에서 요청을 보내면 서버에서 응답하는 형식 (단방향 통신)
  - 응답을 받은 후 Connection이 끊어짐 (Keep Alive로 일정시간동안 Connection 유지가능)
  - HTTPS는 HTTP에 Secure이 강화된 것 (Secure Socket 추가 -> SSL이나 TLS 프로토콜을 통해 세션 데이터를 암호화)


## 2. Socket 통신
  - 클라이언트와 서버가 서로 요청하고 응답받는것이 가능한 형식 (양방향 통신)
  - Connection이 계속 유지 (자주 데이터를 받아야하는 상황이나 실시간 스트리밍, 채팅 등에 유용)
  - Connection이 계속 유지되기 때문에 HTTP에 비해 많은 리소스가 듦 (잦은 데이터 통신이 아닌경우 HTTP 통신이 유리)

</div>
</font>