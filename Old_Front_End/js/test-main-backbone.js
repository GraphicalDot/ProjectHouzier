
$(document).ready(function(){
window.make_request = function make_request(data, algorithm){ url =  window.process_text_url ; return $.post(url, {"text": data, "algorithm": algorithm}) }
App.RootView = Backbone.View.extend({
	//tagName: "fieldset",
	//className: "well-lg plan",
	tagName: "table",
	className: "no-border",
	template: window.template("root"),
	
	initialize: function(){
		var self = this;
		console.log("Root view called")

	},

	render: function(){
		
		this.$el.append(this.template(this));
		
		return this;
	},
	
	events: {
		"click #submit-food": "submitFood",
		"click #submit-service": "submitService",
		"click #submit-cost": "submitCost",
		"click #submit-ambience": "submitAmbience",
		},

	submitFood: function(event){
		$(".main-body").empty()
		event.preventDefault();
		this.submit("food");
	},
	submitService: function(event){
		$(".main-body").empty()
		event.preventDefault();
		bootbox.alert("This word cloud has not been implemented yet!!");
		//this.submit("service");
	},
	submitCost: function(event){
		$(".main-body").empty()
		event.preventDefault();
		this.submit("cost");
	},
	submitAmbience: function(event){
		$(".main-body").empty()
		event.preventDefault();
		this.submit("ambience");
	},

	submit: function(tag){
		console.log("Thus button has been clicked" + tag);	
		bootbox.dialog({
			closeButton: false, 
			message: "<img src='css/images/loading__a.gif'>",
			className: "loadingclass",
		}); 

		console.log("submit button cliked")
		text = $("#searchQuery").val()
		var jqhr = $.post(window.raw_text_processing, {"text": text, "tag": tag})

		jqhr.done(function(data){
			 var subView = new App.WordCloudWith_D3({model: data.result});
			 $(".loadingclass").modal("hide");
		 
			 $.each(data.sentences, function(iter, sentence){
			
			var subView = new App.RootRowView({model: sentence});
			$(".main-body").append(subView.render().el)
				})
			 });
		jqhr.fail(function(){
			bootbox.alert("Either the api or internet connection is not working, Try again later")
			self.$el.removeClass("dvLoading"); 
			event.stopPropagation();
		});
		//var algorithm = $("#appendAlgorithms").find("option:selected").val()
		//this.processText(algorithm)
		},


});

App.RootRowView = Backbone.View.extend({
        tagName: "fieldset",
        className: "well plan each_row",
        template: window.template("root-row"),
        tag: function(){return this.model.tag},
        sentiment: function(){return this.model.sentiment},
        sentence: function(){return this.model.sentence},

        initialize: function(options){
                var self = this;
		
                this.tags = {"food": 1, "service": 2, "ambience": 3, "cost": 4, "null": 5, "overall": 6};
                this.sentiments = {"super-positive": 1, "positive": 2, "neutral": 3, "negative": 4, "super-negative": 5, "mixed": 6};
		this.model = options.model;
	},

        render: function(){
                this.$el.append(this.template(this));
                this.$("#ddpFilter option[value='" + this.tags[this.tag()] + "']").attr("selected", "selected")
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



