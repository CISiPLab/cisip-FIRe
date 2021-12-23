$(document).ready(function(){
    $('form input').change(function () {
      $('form p').text(this.files[0].name + " selected");
    });
    $('#upload').change(function () {
        $('#upload-label').text('File name: ' + this.files[0].name)
    })
  });
