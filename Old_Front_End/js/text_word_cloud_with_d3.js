$(document).ready(function(){

App.SeeWordCloudDateSelectionView = Backbone.View.extend({
	template: window.template("see-word-cloud-date-selection"),
	tag: "form",
	className: "form-horizontal",
	initialize: function(options){
		function newfunction(){
			function another(){
				console.log(this)
			}
			another()
		}
		console.log(this)
		newfunction()
	},

	render: function(){
		this.beforeRender();	
		this.$el.append(this.template(this));
		return this;	
	},

	beforeRender: function(){
		var self = this;
		var jqhr = $.post(window.get_start_date_for_restaurant, {"eatery_id": $("#eateriesList").find('option:selected').attr("id")})	
		jqhr.done(function(data){
			console.log(data.result)
			self.$("#startDate").val(data.result.start)
			self.$("#selectStartDate").val(data.result.start)
			self.$("#endDate").val(data.result.end)
			self.$("#selectEndDate").val(data.result.end)
		});


	}, 

	events: {
		"click #idSubmit": "submit",
		"click #idCancel": "cancel",

	},

	loading_bootbox: function(){
		$(".data_selection").modal("hide");
		bootbox.dialog({ 
			closeButton: false, 
			message: "<img src='css/images/gangam.gif'>",
			className: "loading_dialogue",
		});
	},

	submit: function(event){
		var self = this;
		$(".dynamic_display_word_cloud").empty();	
		event.preventDefault();
		this.$el.addClass("dvLoading");
	
		this.loading_bootbox()


		if ($('#startDate').val() > $('#selectStartDate').val()){
			bootbox.alert("Genius the start date selected should be greater then start date").find('.modal-content').addClass("bootbox-modal-custom-class");
			return

		}
		if ($('#endDate').val() < $('#selectEndDate').val()){

			bootbox.alert("Genius the end date selected should be less then end date").find('.modal-content').addClass("bootbox-modal-custom-class");
			return
		}
	
	
		var jqhr = $.post(window.get_word_cloud, {"eatery_id": $("#eateriesList").find('option:selected').attr("id"),
							"start_date": $('#selectStartDate').val(),
							"end_date": $('#selectEndDate').val(),
		    					"category": $("#wordCloudCategory").find("option:selected").val(),
						})	
		
		
		//On success of the jquery post request
		jqhr.done(function(data){
			var subView = new App.WordCloudWith_D3({model: data.result});
			$(".loading_dialogue").modal("hide");
			});

		//In case the jquery post request fails
		jqhr.fail(function(){
				bootbox.alert("Either the api or internet connection is not working, Try again later")
				self.$el.removeClass("dvLoading");
				event.stopPropagation();
		});

	
	},





	cancel: function(event){
		event.preventDefault();
		$(".data_selection").modal("hide");
	},

});

App.WordCloudWith_D3 = Backbone.View.extend({
	initialize: function(options){
		this.model = options.model;
		console.log(this.model)
		this.render()
	},

	render: function(){
		var w = $(window).width()
		var h = $(window).height()
		
		w.left_margin = 50
		w.right_margin = 50

		h.top_margin = 50
		h.bottom_margin = 50

		function calculateCloud(data){




			console.log("length of the data is" + data.length)
			d3.layout.cloud()
				.size([1200, 1500])
				//.size([w - w.left_margin*2 - w.right_margin*2, h - h.top_margin*2 - h.left_margin*2])
				.words(data)
				.rotate(function() { return ~~(Math.random()*2) * 90;}) // 0 or 90deg
				.fontSize(function(d) { return d.frequency*10; })
				.padding(7)
				.on('end', drawCloud)
				.start();
		}
		function drawCloud(words) {
			d3.select('.dynamic_display').append('svg')
				.attr('width', 1300).attr('height', 1600)
				//.attr('width', w - w.left_margin - w.right_margin)
				//.attr('height', h - h.top_margin - h.left_margin)
				.append('g')
				.attr("width", 1250)
				.attr("height", 1550)
				//.attr("transform", "translate(" + w/2 + "," + h/2 + ")")
				.attr("transform", "translate(" + w/2 + "," + 1.5*h + ")")
				.selectAll('text')
				.data(words)
				.enter().append('text')
				.style('font-size', function(d) { return d.frequency*10 + 'px'; })
				.style('font-family', function(d) { return d.font; })
				.style('fill', function(d, i) { 
					if(d.polarity == "negative"){
						return "red"
					}
					else{
						return "green"
					}
				
				
				})
				.attr('text-anchor', 'middle')
				.attr('transform', function(d) {
					return 'translate(' + [d.x, d.y] + ')rotate(' + d.rotate + ')';
						})
					.text(function(d) { return d.name; });
					}
		
		
		calculateCloud(this.model)
		console.log(this.model.length)
		
		},


});
