	
$(document).ready(function(){
	App = {} ;
	window.App = App ;
	window.template = function(name){ return Mustache.compile($("#"+name+"-template").html()); };
	window.make_request = function make_request(data){ url =  window.process_text_url ; return $.post(url, {"text": data}) }
	window.URL = "http://localhost:8000/"
	
	//window.URL = "http://ec2-54-68-29-37.us-west-2.compute.amazonaws.com:8080/"
	window.raw_text_processing = window.URL + "raw_text_processing";
	window.update_sentence = window.URL + "change_tag_or_sentiment";

});
