
/**
 * Module dependencies.
 */

var express = require('express')
  , routes = require('./routes')
  , http = require('http')
  , path = require('path')
  , fs = require('fs')
  , uuid = require('uuid')
  , yhat = require('yhat')
  , AWS = require('aws-sdk');

AWS.config.credentials = {
    "accessKeyId": process.env.AWS_ACCESS_KEY,
    "secretAccessKey": process.env.AWS_SECRET_KEY,
    "region": process.env.AWS_REGION
}

var s3 = new AWS.S3();
var app = express();
var yh = yhat.init(process.env.YHAT_USERNAME, process.env.YHAT_APIKEY);

// all environments
app.set('port', process.env.PORT || 3000);
app.set('views', __dirname + '/views');
app.set('view engine', 'ejs');
app.use(express.favicon());
app.use(express.logger('dev'));
app.use(express.bodyParser());
app.use(express.methodOverride());
app.use(express.cookieParser('your secret here'));
app.use(express.session());
app.use(app.router);
app.use(require('stylus').middleware(__dirname + '/public'));
app.use(express.static(path.join(__dirname, 'public')));

// development only
if ('development' == app.get('env')) {
  app.use(express.errorHandler());
}

app.get('/', function(req, res) {
    res.render("example");
});

app.get('/entry', function(req, res) {
    res.render("data-entry");
});

app.post('/', function(req, res) {
    var img = req.body.img
      , n = req.body.n;

    var data = img.replace(/^data:image\/\w+;base64,/, "");
    var buf = new Buffer(data, 'base64');
    
    // save locally
    var f = "numbers/" + uuid.v1() + "_digit_" + n + ".png";
    
    // only write to file system if running locally
    if (process.env.NODE_ENV=="production") {
        fs.writeFile(f, buf);
    }

    // upload to s3
    var params = {Bucket: 'demo-datasets', Key: 'digits/' + f, Body: buf};
    s3.putObject(params, function(err, data) {
        if (err) { 
            console.log("erorr saving to s3: " + err)
        }
    });
    
    res.redirect("/");

});

app.post('/predict', function(req, res) {
    var img = req.body.img;
    var data = img.replace(/^data:image\/\w+;base64,/, "");

    yh.predict("DigitRecognizer", 11, {data: data}, function(data) {
        res.send(data);
    });

});

var port = process.env.PORT || 3000;
app.listen(port, function() {
  console.log("Listening on " + port);
});
