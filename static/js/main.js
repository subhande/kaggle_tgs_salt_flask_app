var imagePreview = document.querySelector('#image-preview');
var imageResult = document.querySelector('#image-result');
var input = document.querySelector('#imageUpload');


input.addEventListener('change', function (event) {
    imagePreview.src = URL.createObjectURL(event.target.files[0]);
    imagePreview.style.visibility = "visible";
})


// This funtion sends input to server and in response it receives predicted image

const sendFile = () => {
    var data = new FormData()
    if(!input.files[0])
        return;
    data.append("file", input.files[0])
    
    fetch('/predict', {
    method: 'POST',
    body: data
    }).then((response) => {
        return response.json();
      })
      .then((myJson) => {
        var b64 =  myJson['image'].split("'")[1];
        imageResult.src = 'data:image/png;base64,' + b64;
        imageResult.style.visibility = "visible";
      });
}

function activeUpload() {
    input.click();
}