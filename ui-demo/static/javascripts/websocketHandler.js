
ws_address = "ws://" + document.location.host + document.location.pathname + document.location.search;
// console.log(ws_address)
var socket = new WebSocket(ws_address);
socket.onmessage = function(e){
    var data = JSON.parse(e.data);
    console.log(data);
    if(data.messageType == "text"){
        var bottom = document.getElementById("bottom");
        var newMessageBox = document.createElement('div');
        newMessageBox.className = "left-align";
        newMessageBox.innerHTML = data.content;
        bottom.insertAdjacentElement("beforebegin", newMessageBox);
        bottom.scrollIntoView({inline: 'end'}); 
    }
    else if(data.messageType == "showLossCurve" | data.messageType == "showImage"){
        var bottom = document.getElementById("bottom");
        var newMessageBox = document.createElement('div');
        newMessageBox.className = "center-align";
        var newImgTag = document.createElement('img');
        newImgTag.setAttribute("src", "http://" + document.location.host + "/resource/" + data.filename );
        newMessageBox.insertAdjacentElement("afterbegin", newImgTag);
        bottom.insertAdjacentElement("beforebegin", newMessageBox);
    }
    else if(data.messageType == "done"){
        path = "http://" + document.location.host + "/resource/" + data.filename;
        document.getElementById("state").setAttribute("src", path);
    }
}

socket.onopen = function(e){
    console.log("websocket is connected.");
};
socket.onerror = function(e){
    console.log(e)
};
socket.onclose = function(e){
    console.log("websocket is closed.");
}