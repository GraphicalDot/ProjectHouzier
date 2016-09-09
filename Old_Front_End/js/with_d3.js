
$(document).ready(function(){

window.template = function(name){ return Mustache.compile($("#"+name+"-template").html()); };
window.data_with_coordinates = [{'polarity': 'negative', 'frequency': 1, 'yoyo': 'p', 'name': 'paneer steak', 'coordinates': [243, 236]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'conti dish', 'coordinates': [1052, 32]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'glad i', 'coordinates': [222, 134]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'bank cards', 'coordinates': [847, 247]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'pizza didn', 'coordinates': [407, 194]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'cottage cheese steak', 'coordinates': [318, 384]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'chicken veloute soup', 'coordinates': [522, 439]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'flavour ...', 'coordinates': [996, 202]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'vegan teetotaller orders', 'coordinates': [142, 166]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'italian steak', 'coordinates': [1082, 73]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'peaceful date', 'coordinates': [523, 151]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'anyways', 'coordinates': [406, 388]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'right amount', 'coordinates': [726, 485]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'superb', 'coordinates': [1085, 376]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'watermelon base', 'coordinates': [841, 281]}, 
{'polarity': 'positive', 'frequency': 2, 'name': 'paneer shashlik', 'coordinates': [543, 60]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'great cafe', 'coordinates': [454, 330]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'cheese steak burger', 'coordinates': [167, 498]}, 
{'polarity': 'positive', 'frequency': 1, 'name': '.25 minutes', 'coordinates': [747, 282]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'n high seats', 'coordinates': [957, 190]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'new favorite place', 'coordinates': [947, 279]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'expensive place', 'coordinates': [622, 351]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'spread', 'coordinates': [1043, 222]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'veg quesadillas', 'coordinates': [772, 350]}, 
{'polarity': 'positive', 'frequency': 1, 'name': '% discount', 'coordinates': [269, 151]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'recommended', 'coordinates': [360, 187]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'italian cheese steak burger', 'coordinates': [682, 34]}, 
{'polarity': 'negative', 'frequency': 1, 'name': '..sister concern', 'coordinates': [776, 412]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'pleasant surprise', 'coordinates': [758, 330]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'crisp corn', 'coordinates': [137, 220]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'bucks + taxes', 'coordinates': [353, 470]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'town house caf\xe9', 'coordinates': [645, 225]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'great place', 'coordinates': [657, 80]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'deep flavor', 'coordinates': [893, 472]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'server joins', 'coordinates': [65, 173]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'food lovers', 'coordinates': [99, 367]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'paneer shashlik', 'coordinates': [561, 87]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good company', 'coordinates': [181, 356]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'wow', 'coordinates': [648, 293]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'wall units', 'coordinates': [50, 35]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'unique dish', 'coordinates': [739, 238]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'pure vegetarian', 'coordinates': [296, 380]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'suggestion', 'coordinates': [257, 196]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'dark chocolate mouse needless', 'coordinates': [675, 361]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'red lamps', 'coordinates': [913, 195]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'margarita pizza', 'coordinates': [154, 454]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'tender meat', 'coordinates': [180, 438]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'n g', 'coordinates': [909, 140]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'bhutte ke kebab', 'coordinates': [1050, 44]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'hummus pita', 'coordinates': [70, 429]}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'real gem', 'coordinates': [374, 285]}, 
{'polarity': 'positive', 'frequency': 2, 'name': 'inr', 'coordinates': [847, 164]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good servicegood ambiencetasty foodnice staffi', 'coordinates': [713, 249]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good food', 'coordinates': [802, 227]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'bell pepper', 'coordinates': [453, 270]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'red pasta', 'coordinates': [626, 348]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'great main course', 'coordinates': [416, 467]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'pasta good', 'coordinates': [452, 409]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'mortedela', 'coordinates': [157, 496]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good burger', 'coordinates': [217, 119]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'delhi ncr', 'coordinates': [701, 268]}, 
{'polarity': 'positive', 'frequency': 2, 'name': 'hookah', 'coordinates': [843, 166]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'old techno tracks', 'coordinates': [61, 105]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'connaught', 'coordinates': [74, 351]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'couch', 'coordinates': [58, 168]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'long time', 'coordinates': [249, 22]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'tough leathery slice', 'coordinates': [83, 162]}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'melt burger', 'coordinates': [608, 319]}]



window.data = [{'polarity': 'negative', 'frequency': 1, 'name': 'paneer steak'},
{'polarity': 'positive', 'frequency': 1, 'name': 'conti dish'}, {'polarity': 'positive', 'frequency': 1, 'name': 'glad i'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'bank cards'}, {'polarity': 'positive', 'frequency': 1, 'name': 'pizza didn'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'cottage cheese steak'}, {'polarity': 'negative', 'frequency': 1, 'name': 'chicken veloute soup'}
, {'polarity': 'negative', 'frequency': 1, 'name': 'flavour ...'}, {'polarity': 'positive', 'frequency': 1, 'name': 'vegan teetotaller orders'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'italian steak'}, {'polarity': 'positive', 'frequency': 1, 'name': 'peaceful date'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'anyways'}, {'polarity': 'negative', 'frequency': 1, 'name': 'right amount'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'superb'}, {'polarity': 'negative', 'frequency': 1, 'name': 'watermelon base'}, 
{'polarity': 'positive', 'frequency': 2, 'name': 'paneer shashlik'}, {'polarity': 'positive', 'frequency': 1, 'name': 'great cafe'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'cheese steak burger'}, {'polarity': 'positive', 'frequency': 1, 'name': '.25 minutes'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'n high seats'}, {'polarity': 'positive', 'frequency': 1, 'name': 'new favorite place'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'expensive place'}, {'polarity': 'positive', 'frequency': 1, 'name': 'spread'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'veg quesadillas'}, {'polarity': 'positive', 'frequency': 1, 'name': '% discount'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'recommended'}, {'polarity': 'positive', 'frequency': 1, 'name': 'italian cheese steak burger'}, 
{'polarity': 'negative', 'frequency': 1, 'name': '..sister concern'}, {'polarity': 'positive', 'frequency': 1, 'name': 'pleasant surprise'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'crisp corn'}, {'polarity': 'positive', 'frequency': 1, 'name': 'bucks + taxes'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'town house caf\xe9'}, {'polarity': 'positive', 'frequency': 1, 'name': 'great place'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'deep flavor'}, {'polarity': 'positive', 'frequency': 1, 'name': 'server joins'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'food lovers'}, {'polarity': 'negative', 'frequency': 1, 'name': 'paneer shashlik'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good company'}, {'polarity': 'positive', 'frequency': 1, 'name': 'wow'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'wall units'}, {'polarity': 'positive', 'frequency': 1, 'name': 'unique dish'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'pure vegetarian'}, {'polarity': 'positive', 'frequency': 1, 'name': 'suggestion'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'dark chocolate mouse needless'}, {'polarity': 'negative', 'frequency': 1, 'name': 'red lamps'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'margarita pizza'}, {'polarity': 'positive', 'frequency': 1, 'name': 'tender meat'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'n g'}, {'polarity': 'positive', 'frequency': 1, 'name': 'bhutte ke kebab'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'hummus pita'}, {'polarity': 'negative', 'frequency': 1, 'name': 'real gem'}, 
{'polarity': 'positive', 'frequency': 2, 'name': 'inr'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good servicegood ambiencetasty foodnice staffi'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'good food'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'bell pepper'}, {'polarity': 'positive', 'frequency': 1, 'name': 'red pasta'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'great main course'}, {'polarity': 'positive', 'frequency': 1, 'name': 'pasta good'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'mortedela'}, {'polarity': 'positive', 'frequency': 1, 'name': 'good burger'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'delhi ncr'}, {'polarity': 'positive', 'frequency': 2, 'name': 'hookah'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'old techno tracks'}, {'polarity': 'positive', 'frequency': 1, 'name': 'connaught'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'couch'}, {'polarity': 'positive', 'frequency': 1, 'name': 'long time'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'tough leathery slice'}, {'polarity': 'positive', 'frequency': 1, 'name': 'melt burger'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'college friend\u2019s'}, {'polarity': 'positive', 'frequency': 1, 'name': 'it\u2019s'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'main part'}, {'polarity': 'positive', 'frequency': 1, 'name': 'perfect'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'penny arabiatta'}, {'polarity': 'negative', 'frequency': 1, 'name': 'brain freezer'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'axis'}, {'polarity': 'positive', 'frequency': 1, 'name': 'non veg'}, 
{'polarity': 'positive', 'frequency': 2, 'name': 'nice place'}, {'polarity': 'positive', 'frequency': 1, 'name': 'available..a mixture'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'red lemonade'}, {'polarity': 'negative', 'frequency': 1, 'name': 'terrible time'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'place u'}, {'polarity': 'negative', 'frequency': 1, 'name': 'warehouse cafe ...'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'paneer steak'}, {'polarity': 'positive', 'frequency': 1, 'name': 'juicy burger'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'classic greek salad'}, {'polarity': 'negative', 'frequency': 1, 'name': 'ordinary stuff'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'satisfactory nd'}, {'polarity': 'positive', 'frequency': 1, 'name': 'perfect place'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'overall gud'}, {'polarity': 'positive', 'frequency': 1, 'name': 'huge acreage'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'chilli paneer'}, {'polarity': 'positive', 'frequency': 1, 'name': 'chinese dish'}, 
{'polarity': 'negative', 'frequency': 1, 'name': '... ..'}, {'polarity': 'positive', 'frequency': 1, 'name': 'fresh bun'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'different types'}, {'polarity': 'positive', 'frequency': 1, 'name': 'chilli potatoes'}, 
{'polarity': 'negative', 'frequency': 1, 'name': 'jazzy lit'}, {'polarity': 'positive', 'frequency': 1, 'name': 'culinary experience'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'thai'}, {'polarity': 'positive', 'frequency': 1, 'name': 'ground floor'}, 
{'polarity': 'positive', 'frequency': 1, 'name': 'juicy patty'}, {'polarity': 'positive', 'frequency': 1, 'name': 'loved'}]

window.colors =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254]


App.ButtonView = Backbone.View.extend({
	template: window.template("button"),
	initialize: function(options){
		this.model = options.model;
		console.log(this.model.__parent)
	},

	render: function(){
		 this.$el.append(this.template(this));
		 return this;
	},
	
	events: {
	         "click #hide": "hide", 
	},
	hide: function(event){
		event.preventDefault();
		new_selection = d3.select(this.el)
		console.log(new_selection)
		//d3.selectAll("circle").transition().duration(10250).delay(250).attr("cy", "50%")
		d3.selectAll("circle").transition().duration(10250).delay(function(d, i){
			return i*100	
		})
		.attr("r", function(d, i){
			return d.frequency*100
		})
		d3.selectAll("text").transition().duration(10250).delay(function(d, i){
			return i*100	
		})
		.attr("font-size", function(d, i){
			return d.frequency*10
		})
	},  

});

App.RootView = Backbone.View.extend({

	initialize: function(options){
	
		this.model = options.model
		console.log("Root view called")
		console.log(this.model.el)
		var subView = new App.ButtonView({model: {"__parent": this.el} }); 
		//this.$el.append(subView.render().el);
		//this.__d3_circular_word_cloud();
		//this.__d3_update();
		this.__d3_update_and_transition();
	},

	render: function(){
		return this;
	},
	
	events: {
	},

	__d3_circular_word_cloud: function(){
		var margin = {top: 19.5, right: 19.5, bottom: 19.5, left: 39.5},
		width = 1200 - margin.right,
		height = 500 - margin.top - margin.bottom;

		var xScale = d3.scale.log().domain([300, 1e5]).range([0, width]);
		var yScale = d3.scale.linear().domain([1, 10]).range([height, 0]);
		var xAxis = d3.svg.axis().orient("bottom").scale(xScale).ticks(12, d3.format(",d"));
		var yAxis = d3.svg.axis().scale(yScale).orient("left");


		
		var self = this;
		var w = 1200;
		var h = 700;
		this.__svg = d3.select(this.el).append("svg")
				.attr("width", width + margin.left + margin.right)
				.attr("height", height + margin.top + margin.bottom)
				.style("border", "3px solid yellow")
		this.svg = this.__svg.append("g")
				.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
		
		this.svg.append("g")
			.attr("class", "x axis")
			.attr("transform", "translate(0," + height + ")")
			.call(xAxis);

		this.svg.append("g")
			.attr("class", "y axis")
			.call(yAxis);


		this.svg.selectAll("circle")
				.data(window.data_with_coordinates)
				.enter()
				.append("circle")
				.attr("cx", function(d) {
					        return d.coordinates[0];
						   })
				.attr("cy", function(d){
						return d.coordinates[1];
				      })
				.attr("r", 0)
				.transition()
				.attr("r", function(d, i){
					return d.frequency	
				})

		this.svg.selectAll("circle")
				.data(window.data_with_coordinates)
				.enter()
				.append("circle")
				.attr("cx", function(d) {
					        return d.coordinates[0];
						   })
				.attr("cy", function(d){
						return d.coordinates[1];
				      })
				.attr("r", function(d, i){
					return d.frequency	
				})
				.attr("fill", "rgba(, 0, 128, d.frequency*4)" )
				.attr("stroke", "orange")
				.attr("stroke-width", function(d) {
					return d.frequency*10;
				})
				.attr("opacity", "0.5")
			   .attr("font-family", "sans-serif")
			   .attr("font-size", function(d, i){
			   	return d.frequency*10
			   })
			   .attr("fill", function(d, i){
			   	if(d.polarity == "negative"){
					return "red"
				}
				else{
					return "yellow"
				}
				})	
		this.svg.selectAll("text")
			   .data(window.data_with_coordinates)
			   .enter()
			   .append("text")
			   .text(function(d) {
			   		return d.name;
			   })
			   .attr("x", function(d) {
			   		return d.coordinates[0];
			   })
			   .attr("y", function(d) {
			   		return d.coordinates[1];
			   })
			   .attr("font-family", "sans-serif")
			   .attr("font-size", function(d, i){
				   return d.frequency
				   
			   })
			   .attr("fill", function(d, i){
			   	if(d.polarity == "negative"){
					return "green"
				}
				else{
					return "black"
				}
				})	
			.attr("text-anchor", "middle")
		},

	__d3_update_and_transition: function(){
		var self = this;
		var alphabet = "abcdefghijklmnopqrstuvwxyz".split("");

		var width = 960;
		var height = 500;
		this.svg = d3.select("body").append("svg")
			.attr("width", width)
			.attr("height", height)
			.append("g")
			.attr("transform", "translate(32," + (height / 2) + ")");

		
		function update(data) {
			var text = self.svg.selectAll("text")
				.data(data, function(d) { return d; });
			
			text.attr("class", "update")
				.transition()
				.duration(750)
				.attr("x", function(d, i){ return i*32;});
				
			text.enter().append("text")
					.attr("class", "enter")
					.attr("x", function(d, i){return i*32;})
					.attr("dy", ".35em")
					.attr("y", -60)	
					.style("fill-opacity", 1e-6)	
					.text(function(d){return d;})
					.transition()
					.duration(750)
					.attr("y", 0)
					.style("fill-opacity", 1);

				text.exit()
					.attr("class", "exit")
					.transition()
					.duration(1200)
					.attr("y", 120)
					.style("fill-opacity", 1e-6)
					.remove();
					}
		
		update(alphabet);
		
		setInterval(function() {
			update(shuffle(alphabet)
				.slice(0, Math.floor(Math.random() * 26))
				.sort()
				);
		
			}, 1500);
		
		function shuffle(array) {
			var m = array.length, t, i;
			while (m) {
				i = Math.floor(Math.random() * m--);
				t = array[m], array[m] = array[i], array[i] = t;
			}
			return array;
		}

	},
	__d3_update: function(){
		var self = this;
		var alphabet = "abcdefghijklmnopqrstuvwxyz".split("");

		var width = 960;
		var height = 500;
		this.svg = d3.select("body").append("svg")
			.attr("width", width)
			.attr("height", height)
			.append("g")
			.attr("transform", "translate(32," + (height / 2) + ")");

		
		function update(data) {
			var text = self.svg.selectAll("text")
				.data(data);
			
			text.attr("class", "update");
				
			text.enter().append("text")
					.attr("class", "enter")
					.attr("x", function(d, i) {return i*35; })
					.attr("dy", ".35em");
				
				text.text(function(d) { return d; });
				text.exit().remove();
				}
		
		update(alphabet);
		
		setInterval(function() {
			update(shuffle(alphabet)
				//.slice(0, Math.floor(Math.random() * 26))
				//.sort()
				);
		
			}, 1500);
		
		function shuffle(array) {
			var m = array.length, t, i;
			while (m) {
				i = Math.floor(Math.random() * m--);
				t = array[m], array[m] = array[i], array[i] = t;
			}
			return array;
		}

	},
});
});


