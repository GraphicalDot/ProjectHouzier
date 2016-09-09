/* global $ */
/* global d3 */
function json_maker(input) {
	var output = [];
	var categories = [
		{ 'value': 'terrible', 'color': '#a50f15' }, { 'value': 'poor', 'color': '#de2d26' }, { 'value': 'mix', 'color': '#fb6a4a' },
		{ 'value': 'average', 'color': '#74c476' }, { 'value': 'good', 'color': '#31a354' }, { 'value': 'excellent', 'color': '#006d2c' }];
	categories.forEach(function (category) {
		var obj = { key: category.value, color: category.color, values: [] };
		output.push(obj);
	});
	//in food case, we get an array
	if (Array.isArray(input)) {
		input.forEach(function (key) {
			var dataSet = key;
			output[0].values.push({ "label": dataSet.name, value: -dataSet.terrible ? -dataSet.terrible : 0, total: dataSet.total_sentiments });
			output[1].values.push({ "label": dataSet.name, value: -dataSet.poor ? -dataSet.poor : 0, total: dataSet.total_sentiments });
			output[2].values.push({ "label": dataSet.name, value: +dataSet.mix ? +dataSet.mix : 0, total: dataSet.total_sentiments });
			output[3].values.push({ "label": dataSet.name, value: +dataSet.average ? +dataSet.average : 0, total: dataSet.total_sentiments });
			output[4].values.push({ "label": dataSet.name, value: +dataSet.good ? +dataSet.good : 0, total: dataSet.total_sentiments });
			output[5].values.push({ "label": dataSet.name, value: +dataSet.excellent ? +dataSet.excellent : 0, total: dataSet.total_sentiments });
		});
	}
	else {
		for (var key in input) {
			if (input.hasOwnProperty(key)) {
				var dataSet = input[key];
				output[0].values.push({ "label": key, value: -dataSet.terrible ? -dataSet.terrible : 0, total: dataSet.total_sentiments });
				output[1].values.push({ "label": key, value: -dataSet.poor ? -dataSet.poor : 0, total: dataSet.total_sentiments });
				output[2].values.push({ "label": key, value: +dataSet.mix ? +dataSet.mix : 0, total: dataSet.total_sentiments });
				output[3].values.push({ "label": key, value: +dataSet.average ? +dataSet.average : 0, total: dataSet.total_sentiments });
				output[4].values.push({ "label": key, value: +dataSet.good ? +dataSet.good : 0, total: dataSet.total_sentiments });
				output[5].values.push({ "label": key, value: +dataSet.excellent ? +dataSet.excellent : 0, total: dataSet.total_sentiments });
			}
		}
	}
	return output;
}

function makeGraph() {
	var keys = ['ambience', 'food', 'cost', 'service'];
	var eatery_id= $("#eatery_id").val();
	$.ajax({
		type: "POST",
		url: 'http://localhost:8000/geteatery',
		data: {"__eatery_id": eatery_id},
		success: function (response) {
			var data = response.result
			$.each(keys, function (i, key) {
				nv.addGraph(function () {
					var chart = nv.models.multiBarHorizontalChart()
						.x(function (d) { return d.label })
						.y(function (d) { return d.value })
						.duration(250)
						.margin({ top: 30, right: 20, bottom: 40, left: 120 })
						.groupSpacing(0.818)
						.showControls(false)
						.stacked(true);

					chart.yAxis.axisLabel('Total number of Sentiments');

					d3.select('svg#' + key)
						.datum(json_maker(data[key]))
						.call(chart);

					nv.utils.windowResize(chart.update);
				});
			});
		}
	});
}


document.getElementById('eatery_id').onkeypress = function (e) {
	if (!e) e = window.event;
	var keyCode = e.keyCode || e.which;
	if (keyCode == '13') {
		makeGraph();
		// Enter pressed
		return false;
	}
}
