// code referenced from: https://www.cnblogs.com/jerehedu/p/7839790.html
var canvas = document.getElementById("canvas-seg");
var cvs = canvas.getContext("2d");  
var drawing =false;

const sleep = (milliseconds) => {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
}

function drawingTool() {
    var penWeight = 0;  // pen size  
    var penColor = '';  // pen color 
    canvas.onmousedown = function(e) {  
    // find pen's coordinates
    var bbox = canvas.getBoundingClientRect();
    var start_x = (e.clientX - bbox.left) * (canvas.width / bbox.width);
    var start_y = (e.clientY - bbox.top) * (canvas.height / bbox.height)  
    // begin drawing
    cvs.beginPath();  
    // pen starting point
    cvs.moveTo(start_x, start_y);  
    // set pen properties  
    cvs.lineCap = 'round';  
    cvs.lineJoin ="round";
    // pen color  
    cvs.strokeStyle = penColor;
    // pen size  
    cvs.lineWidth = penWeight; 
    canvas.onmousemove = function(e){  
        // find pen's coordinates  
        var move_x = (e.clientX - bbox.left) * (canvas.width / bbox.width);  
        var move_y = (e.clientY - bbox.top) * (canvas.height / bbox.height)   
        // draw according to mouse path
        cvs.lineTo(move_x, move_y);
        // render
        cvs.stroke(); 
        }
        canvas.onmouseup = function(e){
                // end drawing
                cvs.closePath();
                canvas.onmousemove = null;  
                canvas.onmouseup = null;  
            }  
            canvas.onmouseleave = function(){
                cvs.closePath();
                canvas.onmousemove = null;  
                canvas.onmouseup = null; 
        }
    }
}

function changePenSize(width) {
    cvs.lineWidth = width;
    document.getElementById('penSizeText').innerHTML = width.toString(10); 
}

function changePenColor(pencolor) {
    cvs.strokeStyle = pencolor
}

function clearCanvas() {
    cvs.clearRect(0, 0, 512, 256);
}

function sendToFlask(generator) {
    var imgUrl = canvas.toDataURL("image/png");
        var imageDataB64 = imgUrl.substring(22);

        imgData = {DrawImg:imageDataB64, Generator:generator};
        var senddata = JSON.stringify(imgData);

        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/demo", true);
        xhr.setRequestHeader('content-type', 'application/json');
        xhr.send(JSON.stringify(senddata));

        sleep(2000).then(() => {
            showResult();
        })
}

function showResult() {
    // get canvas
    var cvs = document.getElementById("canvas-gen");
    // create image object
    var imgObj = new Image();
    imgObj.src = "/static/usr_img/gen_img.jpg"+"?timestamp="+new Date().getTime();
    // display image on canvas
    imgObj.onload = function() {
            var ctx = cvs.getContext("2d");
            ctx.drawImage(this, 0, 0, 512, 256);
    }
}

window.onload = function() {this.drawingTool();}