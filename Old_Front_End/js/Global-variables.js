	
$(document).ready(function(){
	App = {} ;
	window.App = App ;
	window.template = function(name){ return Mustache.compile($("#"+name+"-template").html()); };
	window.make_request = function make_request(data){ url =  window.process_text_url ; return $.post(url, {"text": data}) }
	window.URL = "http://localhost:8000/"
	
	//window.URL = "http://ec2-54-68-29-37.us-west-2.compute.amazonaws.com:8080/"
	window.get_start_date_for_restaurant = window.URL + "get_start_date_for_restaurant";
	window.limited_eateries_list = window.URL + "limited_eateries_list";
	window.get_word_cloud = window.URL + "get_word_cloud";
	window.update_sentence = window.URL + "change_tag_or_sentiment";

});
