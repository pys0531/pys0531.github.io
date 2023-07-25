---
title:  <font size="5">Web Server</font>
excerpt: "Web Server"
toc: true
toc_sticky: false
use_math: true
categories:
  - Server
tags:
  - Web Server
  - WAS
  - Web Application Server
  - Nginx
  - Reverse Proxy
  - Gunicorn
  - WSGI
  - Uvicorn
  - ASGI
last_modified_at: 2023-05-02T11:10:00-55:00
---

--------
#### <font size = "4">Web Server</font>
<div markdown = "1">
보통 Web Server는 정적인 컨텐츠와 동적인 컨텐츠를 처리한다. <br>
.html .jpeg .css 등과 같은 정적인 컨텐츠는 Web Server에서 대부분 처리가 되며, 캐싱을 이용해 콘텐츠를 저장하고 다음 요청시 빠르게 보내주게 된다.
또한 Web Server는 동적인 컨텐츠를 받으면 WAS로 전달해주는 역할을 한다. <br>

WAS는 Web Application Server로 동적 컨텐츠를 처리하며, Client의 요청을 받아 DB 조회나 어떤 로직을 처리해야하는 동적인 컨텐츠를 응답하는 서버이다. 주로 DB 서버와 같이 수행된다.<br>
Web Server의 역할도 수행이 가능하지만, 따로 분리하여 사용하는것이 좋다.

Web Server 와 WAS는 분리되어 있으면, 
 1. 과부화를 막을수 있으며
 2. 보안이 강화되고
 3. 로드밸런싱을 위해 여러대의 WAS를 연결해서 사용할수 있다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-07-25-Web Server/Network.png){: .align-center}
<center><font size = "1"> 출처 : https://blog.naver.com/PostView.naver?blogId=seek316&logNo=222410286839 </font></center>
</div>

<br>



#### <font size = "4">Nginx</font>

<div markdown = "1">
-> 동시 접속 처리에 특화된 웹서버<br><br>
Nginx는 Client에서 전달된 요청을 대신 받아서 내부망 sever로 전달하는 Proxy Server (and Web Server) 역할을한다. Proxy(대리자)는 말그대로 대신 전달을 하는 대리자를 뜻한다.<br>

이런 Nginx을 사용하면 장점이 있다.
1. Event Driven 구조로 빠른 속도
   - 유저의 Connection / Request / Close 모든 절차를 이벤트라고 취급
   - 이러한 이벤트를 병렬처리하여 빠른 속도를 갖게하는 구조를 "Event Driven 구조"라고 함
   - Thread Pool을 만들어 오래 걸리는 작업을 할당하고 다른일을 처리
2. 클라이언트 비동기 I/O 처리 방식으로 빠른 응답 시간 보장
   - 높은 성능과 적은 메모리 사용
3. Reverse Proxy로 사용하여 다양한 이점을 얻음
  - Forward Proxy는 Client 뒤에 놓여 Client의 요청이 서버에 직접 전송되지 않고 Proxy서버를 통해 간접적으로 요청하고 응답 받는 방식 (Client의 정보가 감춰짐)
  - Reverse Proxy는 웹서버/WAS 앞에 놓이며, Proxy서버가 내부망서버로부터 데이터를 가져와 응답하는 형식 (서버쪽 정보가 감춰짐) <br>
    -> 최종 형태 : Client -> Internet -> (1차 방화벽) -> DMZ(Reverse Proxy/웹서버) -> (2차 방화벽) -> 내부망(WAS/DB) <br>
    -> 트래픽이 방대하지 않을때는 WAS를 DMZ넣고 서비스를 해도 되지만, WAS는 DB서버랑 연결되어있기 때문에 WAS가 해킹당할 경우 DB서버까지 해킹당할 우려가 있음 <br>

4. Reverse Proxy 이점
 - 로드 밸런싱으로 인해 많은 트래픽에 대한 대처 <br>
   -> 많은 트랙픽에 대한 대처에는 <br>
     -> Scale up: 기존 서버의 성능을 높힘 <br>
     -> Scale out: 여러대의 서버를 두어 트래픽 분산 <br>
   -> Nginx는 scale out을 통해 받는 요청의 load(트래픽)를 고르게 분배해줘 과부화를 막아줌(로드밸런싱) <br>
   -> 서버로 사용할 PC 포트를 여러개 적어주기만 하면됨, 서버PC에는 소프트웨어만 설치하면 끝나므로 시간과 비용 절약 <br>
 - Caching을 통해 컨텐츠를 저장하고 다음 요청시 빠르게 응답
 - Client에 응답을 Reverse Proxy가 해줌으로써, 응답이 서버에서 시작이 아닌 Reverse Proxy에서 시작된것 처럼 보여짐
 - 이외 SSL 인증서, 압축, 보안 등 다양한 이점이 있음
</div>

<br>

#### <font size = "3">Gunicorn(WSGI) / Uvicorn(ASGI)</font>
<div markdown = "1">
 - WSGI(Web Server Gateway Interface) : Python에서 앱이 웹서버와 통신하기 위한 표준 인터페이스
 - ASGI(Asynchronous Server Gateway Interface) : WSGI와 호환되며, 여러 서버 및 어플리케이션 프레임 워크를 통해 비동기 및 동기 모두 제공
 - Gunicorn : WSGI 웹 애플리케이션을 실행하는 서버
 - Uvicorn : ASGI 웹 애플리케이션을 실행하는 서버
   - Uvicorn은 여러 작업자(Worker)를 통해 애플리케이션을 구동하는것이 어렵다. 기본적으로 하나의 작업자에서만 실행되므로, 여러 CPU코어를 활용하는 등의 병렬처리를 위해서는 다른방법이 필요
   - Gunicorn(WSGI) / Uvicorn(ASGI)를 같이 사용할 경우
     - Gunicorn은 프로세스 매니저 / Uvicorn은 Worker 프로세스로 사용됨, 즉, Gunicorn은 Master 프로세스 하나를 띄우고 / Uvicorn은 단일 Worker프로세스가 미리 여러개 띄워 요청을 처리하는 구조
     - Gunicorn은 Python 웹 애플리케이션을 병렬로 처리할 수 있는 Pre-fork 워커 모델을 지원함. 이를 통해 Uvicorn과 같이 사용하면, FastAPI 애플리케이션을 여러 작업자로 분산하여 처리하고, 동시에 다수의 HTTP 요청을 빠르게 처리할 수 있음.
     - 명령어 : gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -> 4개의 Worker와 Uvicorn Worker를 사용하겠다는 의미


![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-07-25-Web Server/fastapi.png){: .align-center}
<center><font size = "1"> 출처 : https://breezymind.com/start-asgi-framework/ </font></center>


</div>

