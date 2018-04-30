const express = require('express')
const app = express()
const port = 3000
var http = require("http").Server(app);
var io = require("socket.io")(http); 
var jsonfile = require('jsonfile');
var fs = require('fs');

app.use(express.static(__dirname + "/public" ));

app.get('/', function(req, res) {
  res.redirect('index.html');
});
 
io.on('connection', function(socket){
	socket.on('review',function(data){

		var newjson = {
            review: data
        }
        var json = JSON.stringify(newjson);
        var fs = require('fs');
        fs.writeFile('review.txt', json, 'utf8');    
	});

	socket.on('getrating',function(data){
		var obj = JSON.parse(fs.readFileSync('predicted.txt', 'utf8')); 
		console.log(obj.stars)
		socket.emit('stars', obj.stars);  
	});


}) 

http.listen(port,function(){
    console.log('Server listening through port %s', port);
}); 
 