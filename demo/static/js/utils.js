function showSample() {
    // get canvas
    var cvs = document.getElementById("canvas-seg");
    // create image object
    var imgObj = new Image();
    imgObj.src = "/static/usr_img/seg_map.png"+"?timestamp="+new Date().getTime();
    // display image on canvas
    imgObj.onload = function() {
            var ctx = cvs.getContext("2d");
            ctx.drawImage(this, 0, 0, 512, 256);
    }
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

window.onload = function() {this.showSample(); this.showResult();}
