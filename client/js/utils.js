let model;
let IMAGE_WIDTH = 300;

URL_PATH = 'http://0.0.0.0:5000/predict'
URL_PATH = 'http://127.0.0.1:5000//predict'

function toggleDiv(){
  $('#image-frame').toggle()
}

function chooseCNN() {
  $('.model-button').removeClass('active')
  $('#cnn').addClass('active');
}

function chooseSVM() {
  $('.model-button').removeClass('active')
  $('#svm').addClass('active');
}

function loadImageLocal() {
  if(document.getElementById("roi").getElementsByTagName("img")[0])
    document.getElementById("roi").getElementsByTagName("img")[0].remove();
  if(document.getElementById("rmb").getElementsByTagName("img")[0])
    document.getElementById("rmb").getElementsByTagName("img")[0].remove();
  document.getElementById("prediction").innerHTML = ""
  document.getElementById("step").style.display = "none";
  document.getElementById("select-file-box").style.display = "table-cell";
  document.getElementById("predict-box").style.display = "table-cell";
  // document.getElementById("prediction").innerHTML = "Click predict to find my label!";
  renderImage();
};

function renderImage() {
  var reader = new FileReader();
  reader.onload = function(event) {
    let output = document.getElementById('test-image');
  	output.src = reader.result;
  	output.width = IMAGE_WIDTH;
  }
  
  if(event.target.files[0]){
	reader.readAsDataURL(event.target.files[0]);
  }
}

function predict(){
  console.log($('.active').val())
}

async function predictImage(){
    if(document.getElementById("roi").getElementsByTagName("img")[0])
      document.getElementById("roi").getElementsByTagName("img")[0].remove();
    if(document.getElementById("rmb").getElementsByTagName("img")[0])
      document.getElementById("rmb").getElementsByTagName("img")[0].remove();
    model_type = $('.active').val()
    document.getElementById("progress-box").style.display = "block";
    let output = document.getElementById('test-image').src;
    imgbase64 = output.split(',')[1];
    console.log(model_type)
    const body = { imgbase64: imgbase64, model: model_type };
    console.log(body)
    const config = {
      method: 'POST',
      headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(body)
    }
    const response = await fetch(URL_PATH, config)
    const json = await response.json()
    if (response.ok) {
        document.getElementById("progress-box").style.display = "none";
        console.log(json)
        document.getElementById("image-frame").style.display = "none";
        document.getElementById("step").style.display = "block";
        document.getElementById("prediction").innerHTML = "License plate: <b>" +json['lp'].join("") + "</b>"+" in <b style='color: black;font-size:12px'>" + json['time'] +"s<b>";
        // document.getElementById("fname").value = json['time'];
        if(!json['rmb'] & !json['roi']) {
          document.getElementById("step").style.display = "none";
        }

        if(json['rmb'])
        {
          var image = new Image();
          image.src = 'data:image/png;base64,'+json['rmb'];
          document.getElementById("rmb").appendChild(image);
          document.getElementById("rmb").getElementsByTagName("img")[0].width =200
        }

        if(json['roi']) {
          var image = new Image();
          image.src = 'data:image/png;base64,'+json['roi'];
          document.getElementById("roi").appendChild(image);
          document.getElementById("roi").getElementsByTagName("img")[0].width =200
        }
        

        return response
    } else {
        //
    }

}
