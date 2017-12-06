if (window.addEventListener) {
    window.addEventListener('load', function () {
        var ws = new WebSocket("ws://104.196.148.187:8000/char");
        var canvas, context, tool;

        function init() {
            canvas = document.getElementById('imageView');
            context = canvas.getContext('2d');
            tool = new Pencil();

            canvas.addEventListener('mousedown', ev_canvas, false);
            canvas.addEventListener('mousemove', ev_canvas, false);
            canvas.addEventListener('mouseup', ev_canvas, false);
        }
	
        $('#addSpace').click(function () {
            var $output = $('#outputText');
            var text = $output.text();
            $output.text(text + ' ');
        });

        ws.onmessage = function (evt) {
            var $output = $('#outputText');
            var text = $output.text();
            var newText = text +  evt.data;
            $output.text(newText);
        };

        function Pencil() {
            var tool = this;
            this.started = false;
            context.lineWidth = 1;
            context.shadowBlur = 0;
            context.shadowColor = "black";
            context.lineJoin = "round";
            context.lineCap = "round";

            this.mousedown = function (ev) {
                if (ws.readyState === WebSocket.OPEN) {
                    context.beginPath();
                    context.moveTo(ev._x, ev._y);
                    tool.started = true;

                    if (context.timeout !== undefined) {
                        var time = (new Date()).getTime() / 1000;
                        var diff = time - context.time;
                        if (diff < 1) {
                            window.clearTimeout(context.timeout);
                        }
                    }
                }
            };

            this.mousemove = function (ev) {
                if (ws.readyState === WebSocket.OPEN && tool.started) {
                    context.lineTo(ev._x, ev._y);
                    context.stroke();
                }
            };

            this.mouseup = function (ev) {
                function timeout(n) {
                    if(context.timeout){
                        clearTimeout(context.timeout);
                        delete context.timeout;
                    }

                    context.timeout = setTimeout(function () {
                        var raw = context.getImageData(0, 0, 336, 336);
                        ws.send(JSON.stringify({ data : Array.from(raw.data) }));
                        context.clearRect(0, 0, 336, 336);
                    }, n);
                }

                if (ws.readyState === WebSocket.OPEN && tool.started) {
                    tool.mousemove(ev);
                    tool.started = false;

                    if (context.timeout !== undefined) {
                        context.time = (new Date()).getTime() / 1000;
                    }
                    timeout(1000);
                }
            };
        }

        function ev_canvas(ev) {
            if (ev.layerX || ev.layerX === 0) { // Firefox
                ev._x = ev.layerX;
                ev._y = ev.layerY;
            } else if (ev.offsetX || ev.offsetX === 0) { // Opera
                ev._x = ev.offsetX;
                ev._y = ev.offsetY;
            }

            var func = tool[ev.type];
            if (func) {
                func(ev);
            }
        }

        init();

    }, false);
}
