/* global Materialize */
/* global $ */
var url = "http://localhost:8000/texttokenization";


function clearTable() {
	$(".result-row table tbody").empty();
}

function fillTable(tableData) {
	clearTable();
	var tableString = '';
	$.each(tableData, function (i, data) {
		tableString = '<tr><td>' + data.sentence + '</td><td>' + data.sentiment + '</td><td>' + data.category + '</td><td>' + data.sub_category + '</td><td>';
		if( typeof data.noun_phrase_or_place === 'string' ) {
			tableString += '<div class="chip">'+data.noun_phrase_or_place+'</div>';
		} else {
			$.each(data.noun_phrase_or_place, function (j, phrase) {
				tableString += '<div class="chip">' + phrase + '</div>';
			});
		}
		tableString += '</td></tr>';
		$(".result-row table tbody").append(tableString);
	});
	$(".result-card").removeClass('hide');
	$('.loader').addClass('hide');
}

$('document').ready(function () {
	$('#submitButton').click(function (e) {
		e.preventDefault();
		if (!$("#sentenceBox").val()) {
			Materialize.toast('Enter a Sentence First', 3000);
			return;
		}
		$('.loader').removeClass('hide');
		var xhrRequest = $.post(url, { text: $("#sentenceBox").val() });
		xhrRequest.done(function (data) {
			if (data.success) {
				fillTable(data.result);
			} else {
				$('.loader').addClass('hide');
				Materialize.toast('Ajax Request returned error', 3000);
			}
		}).fail(function (msg) {
			$('.loader').addClass('hide');
			Materialize.toast('Server request failed. Populating Dummy Data', 3000);
			// var dummyData = { "error": false, "success": true, "result": [{ "category": "food", "noun_phrase_or_place": ["positives end", "chicken wings", "veg croquettes"], "sentence": "the veg croquettes and chicken wings were deliciousand that's where the positives end .", "sentiment": "good", "sub_category": "dishes" }, { "category": "food", "noun_phrase_or_place": ["chicken satay"], "sentence": "the chicken satay and martini were badafter about an hour of extreme discomfort we decided to leave .", "sentiment": "average", "sub_category": "dishes" }, { "category": "place", "noun_phrase_or_place": ["gurgaon"], "sentence": "now imagine an overcrowded place in gurgaon's summer where the ac isn't working .", "sentiment": "poor", "sub_category": "none" }, { "category": "service", "noun_phrase_or_place": "none", "sentence": "the hostess and waiters were warm and 2", "sentiment": "poor", "sub_category": "staff" }, { "category": "service", "noun_phrase_or_place": "none", "sentence": "the hostess had some polite words when we told her ,", "sentiment": "average", "sub_category": "service-null" }, { "category": "ambience", "noun_phrase_or_place": "none", "sentence": "they played some music", "sentiment": "poor", "sub_category": "music" }, { "category": "ambience", "noun_phrase_or_place": "none", "sentence": "but the noise from the crowd drowned it .", "sentiment": "poor", "sub_category": "crowd" }, { "category": "overall", "noun_phrase_or_place": "none", "sentence": ":1. Overcrowded .", "sentiment": "poor", "sub_category": "none" }, { "category": "overall", "noun_phrase_or_place": "none", "sentence": "there was hardly any place stand , let alone move .", "sentiment": "poor", "sub_category": "none" }, { "category": "overall", "noun_phrase_or_place": "none", "sentence": "never going back", "sentiment": "good", "sub_category": "none" }] };
			// fillTable(dummyData.result);
		});
	});
});
