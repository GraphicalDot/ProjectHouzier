$(document).ready(function(){
App.SeeWordCloudDateSelectionView = Backbone.View.extend({
	template: window.template("see-word-cloud-date-selection"),
	tag: "form",
	className: "form-horizontal",
	initialize: function(options){
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

	submit: function(event){
		var self = this;
		$(".dynamic_display_word_cloud").empty();	
		event.preventDefault();
		this.$el.addClass("dvLoading");
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
		jqhr.done(function(data){
			$.each(data.result, function(iter, noun_phrase_dict){
				var subView = new App.SeeWordCloudDateMainView({model: noun_phrase_dict});
				$(".dynamic_display_word_cloud").append(subView.render().el);	
				$(".dynamic_display_word_cloud").append('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;');	
			});
			self.$el.removeClass("dvLoading");
		
			});
		jqhr.fail(function(){
				bootbox.alert("Either the api or internet connection is not working, Try again later")
				self.$el.removeClass("dvLoading");
				event.stopPropagation();
		});

	
	},





	cancel: function(event){
		event.preventDefault();
		this.remove();
	},

});

App.SeeWordCloudDateMainView = Backbone.View.extend({
	template: Mustache.compile('{{nounPhrase}}'),
	tagName: "a",
	size: function(){return this.model.frequency*10},
	nounPhrase: function(){return this.model.name},
	polarity: function(){return this.model.polarity},
	

	initialize: function(options){
		this.model = options.model;
		console.log(this.size())
		console.log(this.polarity())
		
	},

	render: function(){
		this.$el.append(this.template(this));
		this.afterRender();
		return this;
	},

	afterRender: function(){
		var self = this;
		this.$el.attr({href: "#", rel: this.size()});
		this.$el.css({"font-size": this.size()})
		if(self.polarity() == "negative"){
			console.log(" Negative polkrariy aaying");	
			self.$el.css({"color": "red"})
		}
	},

});

});

