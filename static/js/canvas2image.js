function getImageFromCanvas(){
  const isChrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
  if (!isChrome) {
    alert('This sample only works on Chrome');
  }
  const canvas = document.getElementById("canvas");
  const prediction = document.getElementById("prediction");
  const img = canvas.toDataURL("image/png");
  const childImg = document.createElement('img')
  childImg.src = img
  prediction.appendChild(childImg);
  document.getElementById('canvasImg').value = img;
}
