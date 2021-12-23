$(document).ready(function(){
    $('form input').change(function () {
      $('form p').text(this.files[0].name + " selected");
    });
    $('#upload').change(function () {
        $('#upload-label').text('File name: ' + this.files[0].name)
    })
  });

// const input = document.getElementById('upload');
// const infoArea = document.getElementById('upload-label');
//
// input.addEventListener( 'change', showFileName );
// function showFileName( event ) {
//     let input = event.srcElement;
//     let fileName = input.files[0].name;
//     infoArea.textContent = 'File name: ' + fileName;
// }