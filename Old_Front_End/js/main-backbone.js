
$(document).ready(function(){
window.make_request = function make_request(data, algorithm){ url =  window.process_text_url ; return $.post(url, {"text": data, "algorithm": algorithm}) }
App.RootView = Backbone.View.extend({
	//tagName: "fieldset",
	//className: "well-lg plan",
	tagName: "table",
	className: "table table-hover borderless scroll",
	template: window.template("root"),
	
	initialize: function(){
		var self = this;
		var jqhr = $.get(window.limited_eateries_list)	
		
		jqhr.done(function(data){
			console.log(data)
			console.log(data.result)
			if (data.error == false){
				$.each(data.result, function(iter, eatery){
					var subView = new App.AppendRestaurants({"eatery": eatery});
					self.$(".append_eatery").append(subView.render().el);	
				})
						}
			else{
				bootbox.alert(data.error)	
			}
		//	self.$(".append_eatery").append('<hr><hr><tr><td></td></tr><tr><td><b>Select Sentiment</b></td></tr><tr><td><label class="checkbox"><input type="checkbox" value="1" id="positive" class="sentiment">positive</label></td></tr><tr><td><label class="checkbox"><input type="checkbox" value="0" id="negative" class="sentiment">negative</label></td></tr>')	
		})
		
		jqhr.fail(function(data){
				
				bootbox.alert(data.error)
		});
		console.log("Root view called")
		$(".side-bar").html(this.render().el);
	},

	render: function(){
		
		this.$el.append(this.template(this));
		this.$el.css({
			'height' :"200px",
			'overflow-y': "scroll", 

		})
		return this;
	},
	
	events: {
	
		"click #seeWordCloud": "seeWordCloud",
		"change .change_eatery_id": "ClickEatery",
		//"change .sentiment": "ChaneSentiment",
		},

	ChaneSentiment: function(event){
		event.preventDefault();
		var id = $(event.currentTarget).attr("id")
		console.log("id");
	},


	ClickEatery: function(event){
		event.preventDefault();
		var id = $(event.currentTarget).attr("id")
		$.each($(":checkbox"), function(iter, __checkbox){
			if (__checkbox.id != id){
				$("#" + __checkbox.id).prop("checked", false);


			}
		
		});
		
		
		console.log(id)
		if ($(":checkbox:checked").length == 1){
			//$(".sentiment").prop("checked", true)
			this.seeWordCloud(id);
		};
	},



	seeWordCloud: function(eatery_id){
		$(".main-body").empty()
		
			
		var subView = new App.SeeWordCloudDateSelectionView();
		bootbox.dialog({
			"title": "Word Cloud: " + $(":checkbox:checked").val() ,
			"message": subView.render().el,
			"animate": true,
			"closeButton": true,
			"className": "data_selection",
			})
	},




});


App.AppendRestaurants = Backbone.View.extend({
	template: window.template("append-restaurant"),
	tagName: "tr",
	eatery_name: function(){return this.model.eatery.eatery_name},
	eatery_id: function(){return this.model.eatery.eatery_id},
	reviews: function(){return this.model.eatery.reviews},
	area: function(){return this.model.eatery.area_or_city},
	initialize: function(options){
		this.model = this.options;
		console.log(this.eatery_name());
		console.log(this.reviews());
	},

	render: function(){
		this.$el.append(this.template(this));
		this.$(".change_eatery_id").hovercard({
			detailsHTML: '<p><b>' + "Reviews:" + this.reviews() + '</b>'+ '<br>'+ '<b> City: '+ this.area() + '</b></p>',
			width: 90,
			openOnTop: true,
			height: 30,
		});
		return this;	
	
	},


});

App.RootRowView = Backbone.View.extend({
        tagName: "fieldset",
        className: "well plan each_row",
        template: window.template("root-row"),
        sentiment: function(){return this.model.sentiment},
        sentence: function(){return this.model.sentence},

        initialize: function(options){
                var self = this;
		
                this.sentiments = {"super-positive": 1, "positive": 2, "neutral": 3, "negative": 4, "super-negative": 5, "mixed": 6};
		this.model = options.model;
	},

        render: function(){
                this.$el.append(this.template(this));
                this.$("#ddpFiltersentiment option[value='" + this.sentiments[this.sentiment()] + "']").attr("selected", "selected")
		return this;
        },

        events: {
                    "change #ddpFilter" : "changeTag",
                    "change #ddpFiltersentiment" : "changeSentiment",
		    },


	changeSentiment: function(event){
                var self = this;
                event.preventDefault()
                sentence = self.sentence();
                changed_polarity = self.$('#ddpFiltersentiment option:selected').text();
                console.log(self.sentence())
                console.log(changed_polarity)
		var jqhr = $.post(window.update_sentence, {"sentence": sentence, "value": changed_polarity, "whether_allowed": false})
                jqhr.done(function(data){
                        console.log(data.success)
                        if (data.success == true){
                                bootbox.alert(data.messege)
                                }
                        else {
                                bootbox.alert(data.messege)
                                }
                        })

                jqhr.fail(function(){
                        bootbox.alert("Either the api or internet connection is not working, Try again later")
                                })
        
				},


	changeTag: function(event){
                var self = this;
                event.preventDefault()
                changed_tag = self.$('#ddpFilter option:selected').text();
                sentence = self.sentence();

                console.log(self.sentence())
                console.log(changed_tag)
		var jqhr = $.post(window.update_sentence, {"sentence": sentence, "value": changed_tag, "whether_allowed": false})
                jqhr.done(function(data){
                        if (data.success == true){
                                bootbox.alert(data.messege)
                        }
                        else {
                                bootbox.alert(data.messege)
                                }
                        })

                jqhr.fail(function(){
                                bootbox.alert("Either the api or internet connection is not working, Try again later")
                        })
        },

});
});



